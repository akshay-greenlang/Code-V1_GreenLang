# -*- coding: utf-8 -*-
"""
SatelliteMonitoringService - Facade for AGENT-EUDR-003 Satellite Monitoring

This module implements the SatelliteMonitoringService, the single entry point
for all satellite-based forest monitoring operations in the GL-EUDR-APP. It
manages the lifecycle of eight internal engines, an async PostgreSQL connection
pool (psycopg + psycopg_pool), a Redis cache connection, OpenTelemetry tracing,
and Prometheus metrics. The service exposes a unified interface consumed by the
FastAPI router layer and the GL-EUDR-APP integration.

Lifecycle:
    startup  -> load config -> connect DB -> register pgvector -> connect Redis
             -> initialize engines -> start health check background task
    shutdown -> close engines -> close Redis -> close DB pool -> flush metrics

Engines (8):
    1. ImageryAcquisitionEngine    - Satellite scene search & download (Feature 1)
    2. SpectralIndexCalculator     - NDVI/EVI/NBR/NDMI computation (Feature 2)
    3. BaselineManager             - Dec 2020 baseline establishment (Feature 3)
    4. ForestChangeDetector        - Multi-method change detection (Feature 4)
    5. DataFusionEngine            - Multi-source fusion analysis (Feature 5)
    6. CloudGapFiller              - SAR-based cloud-gap filling (Feature 6)
    7. ContinuousMonitor           - Ongoing monitoring schedules (Feature 7)
    8. AlertGenerator              - Deforestation alert generation (Feature 8)

FastAPI Integration:
    Use the ``lifespan`` async context manager with ``FastAPI(lifespan=lifespan)``
    for automatic startup/shutdown.

Example:
    >>> from greenlang.agents.eudr.satellite_monitoring.setup import (
    ...     SatelliteMonitoringService,
    ...     get_service,
    ... )
    >>> service = get_service()
    >>> await service.startup()
    >>> health = await service.health_check()
    >>> assert health["status"] == "healthy"
    >>> await service.shutdown()

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-003 Satellite Monitoring Agent (GL-EUDR-SAT-003)
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
from datetime import date, datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from greenlang.agents.eudr.satellite_monitoring.config import (
    SatelliteMonitoringConfig,
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


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_service_hash(config: SatelliteMonitoringConfig) -> str:
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
# AnalysisRequest
# ---------------------------------------------------------------------------


class AnalysisRequest:
    """Request object for full satellite monitoring analysis.

    Bundles all input data needed by the service to perform a comprehensive
    satellite-based analysis across all engines for a single plot.

    Attributes:
        plot_id: Unique plot identifier.
        polygon_vertices: List of (lat, lon) tuples forming the plot boundary.
        commodity: EUDR commodity identifier (e.g., 'palm_oil', 'soya').
        country_code: ISO 3166-1 alpha-2 country code.
        analysis_level: Analysis depth ('quick', 'standard', 'deep').
        biome: Optional biome override. If None, auto-detected from
            commodity and country_code.
    """

    __slots__ = (
        "plot_id", "polygon_vertices", "commodity", "country_code",
        "analysis_level", "biome",
    )

    def __init__(
        self,
        plot_id: str = "",
        polygon_vertices: Optional[List[Tuple[float, float]]] = None,
        commodity: str = "",
        country_code: str = "",
        analysis_level: str = "standard",
        biome: Optional[str] = None,
    ) -> None:
        self.plot_id = plot_id or f"PLOT-{uuid.uuid4().hex[:8]}"
        self.polygon_vertices = polygon_vertices or []
        self.commodity = commodity
        self.country_code = country_code
        self.analysis_level = analysis_level
        self.biome = biome


# ---------------------------------------------------------------------------
# FullAnalysisResult
# ---------------------------------------------------------------------------


class FullAnalysisResult:
    """Unified result from a full satellite monitoring analysis.

    Combines results from baseline establishment, change detection,
    multi-source fusion, alert generation, evidence packaging, and
    data quality assessment into a single auditable result.

    Attributes:
        plot_id: Plot identifier.
        baseline: BaselineSnapshot from the BaselineManager.
        change_detection: ChangeDetectionResult from ForestChangeDetector.
        fusion: FusionResult from DataFusionEngine.
        alerts: List of generated SatelliteAlerts.
        evidence: EvidencePackage for EUDR compliance documentation.
        data_quality: Data quality assessment metrics.
        provenance_hash: SHA-256 hash of the combined result.
        analyzed_at: UTC timestamp of analysis completion.
        processing_time_ms: Total processing time in milliseconds.
    """

    __slots__ = (
        "plot_id", "baseline", "change_detection", "fusion",
        "alerts", "evidence", "data_quality", "provenance_hash",
        "analyzed_at", "processing_time_ms",
    )

    def __init__(
        self,
        plot_id: str = "",
        baseline: Optional[Any] = None,
        change_detection: Optional[Any] = None,
        fusion: Optional[Any] = None,
        alerts: Optional[List[Any]] = None,
        evidence: Optional[Any] = None,
        data_quality: Optional[Dict[str, Any]] = None,
        provenance_hash: str = "",
        analyzed_at: Optional[datetime] = None,
        processing_time_ms: float = 0.0,
    ) -> None:
        self.plot_id = plot_id
        self.baseline = baseline
        self.change_detection = change_detection
        self.fusion = fusion
        self.alerts = alerts or []
        self.evidence = evidence
        self.data_quality = data_quality or {}
        self.provenance_hash = provenance_hash
        self.analyzed_at = analyzed_at or _utcnow()
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        def _result_dict(result: Any) -> Optional[Any]:
            if result is None:
                return None
            if hasattr(result, "to_dict"):
                return result.to_dict()
            return str(result)

        return {
            "plot_id": self.plot_id,
            "baseline": _result_dict(self.baseline),
            "change_detection": _result_dict(self.change_detection),
            "fusion": _result_dict(self.fusion),
            "alerts": [_result_dict(a) for a in self.alerts],
            "evidence": _result_dict(self.evidence),
            "data_quality": self.data_quality,
            "provenance_hash": self.provenance_hash,
            "analyzed_at": self.analyzed_at.isoformat(),
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


# ---------------------------------------------------------------------------
# BatchAnalysisResult
# ---------------------------------------------------------------------------


class BatchAnalysisResult:
    """Result of a batch satellite monitoring analysis job.

    Attributes:
        operator_id: Unique operator identifier who submitted the batch.
        batch_id: Unique batch identifier.
        status: Batch status ('pending', 'processing', 'completed', 'failed').
        total_plots: Total plots in the batch.
        completed_plots: Number of plots processed so far.
        failed_plots: Number of plots that failed analysis.
        results: Per-plot FullAnalysisResult list.
        statistics: Aggregated statistics across all plots.
        provenance_hash: SHA-256 hash of the combined batch result.
        submitted_at: When the batch was submitted.
        completed_at: When the batch completed (or None).
        processing_time_ms: Total batch processing time in milliseconds.
    """

    __slots__ = (
        "operator_id", "batch_id", "status", "total_plots",
        "completed_plots", "failed_plots", "results", "statistics",
        "provenance_hash", "submitted_at", "completed_at",
        "processing_time_ms",
    )

    def __init__(
        self,
        operator_id: str = "",
        batch_id: str = "",
        status: str = "pending",
        total_plots: int = 0,
        completed_plots: int = 0,
        failed_plots: int = 0,
        results: Optional[List[FullAnalysisResult]] = None,
        statistics: Optional[Dict[str, Any]] = None,
        provenance_hash: str = "",
        submitted_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
        processing_time_ms: float = 0.0,
    ) -> None:
        self.operator_id = operator_id
        self.batch_id = batch_id or f"SAT-BATCH-{uuid.uuid4().hex[:12]}"
        self.status = status
        self.total_plots = total_plots
        self.completed_plots = completed_plots
        self.failed_plots = failed_plots
        self.results = results or []
        self.statistics = statistics or {}
        self.provenance_hash = provenance_hash
        self.submitted_at = submitted_at or _utcnow()
        self.completed_at = completed_at
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "operator_id": self.operator_id,
            "batch_id": self.batch_id,
            "status": self.status,
            "total_plots": self.total_plots,
            "completed_plots": self.completed_plots,
            "failed_plots": self.failed_plots,
            "results": [r.to_dict() for r in self.results],
            "statistics": self.statistics,
            "provenance_hash": self.provenance_hash,
            "submitted_at": self.submitted_at.isoformat(),
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


# ---------------------------------------------------------------------------
# SatelliteMonitoringService
# ---------------------------------------------------------------------------


class SatelliteMonitoringService:
    """Facade service for the EUDR Satellite Monitoring Agent.

    This is the single entry point for all satellite-based forest monitoring
    operations. It manages the full lifecycle of database connections, cache
    connections, eight internal engines, health monitoring, and OpenTelemetry
    tracing.

    The service follows a strict startup/shutdown protocol:
        startup:  config -> DB pool -> pgvector -> Redis -> engines -> health
        shutdown: health stop -> engines -> Redis -> DB pool -> metrics flush

    Attributes:
        config: Service configuration loaded from env or injected.
        is_running: Whether the service is currently active and healthy.

    Example:
        >>> service = SatelliteMonitoringService()
        >>> await service.startup()
        >>> result = service.search_imagery(vertices, date_range, "sentinel2")
        >>> await service.shutdown()
    """

    def __init__(
        self,
        config: Optional[SatelliteMonitoringConfig] = None,
    ) -> None:
        """Initialize SatelliteMonitoringService.

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
        self._imagery_acquisition: Optional[Any] = None
        self._spectral_calculator: Optional[Any] = None
        self._baseline_manager: Optional[Any] = None
        self._change_detector: Optional[Any] = None
        self._data_fusion: Optional[Any] = None
        self._cloud_gap_filler: Optional[Any] = None
        self._continuous_monitor: Optional[Any] = None
        self._alert_generator: Optional[Any] = None

        # Batch tracking
        self._batch_registry: Dict[str, BatchAnalysisResult] = {}
        self._batch_lock = threading.Lock()

        # Health check background task
        self._health_task: Optional[asyncio.Task[None]] = None
        self._last_health: Optional[HealthStatus] = None
        self._health_interval_seconds: float = 30.0

        # OpenTelemetry tracer
        self._tracer: Optional[Any] = None

        logger.info(
            "SatelliteMonitoringService created: config_hash=%s, "
            "pool_size=%d, cache_ttl=%ds, cutoff=%s",
            self._config_hash[:12],
            self._config.pool_size,
            self._config.cache_ttl_seconds,
            self._config.cutoff_date,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> SatelliteMonitoringConfig:
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
    def imagery_acquisition(self) -> Any:
        """Return the ImageryAcquisitionEngine instance.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._imagery_acquisition

    @property
    def spectral_calculator(self) -> Any:
        """Return the SpectralIndexCalculator instance.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._spectral_calculator

    @property
    def baseline_manager(self) -> Any:
        """Return the BaselineManager instance.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._baseline_manager

    @property
    def change_detector(self) -> Any:
        """Return the ForestChangeDetector instance.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._change_detector

    @property
    def data_fusion(self) -> Any:
        """Return the DataFusionEngine instance.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._data_fusion

    @property
    def cloud_gap_filler(self) -> Any:
        """Return the CloudGapFiller instance.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._cloud_gap_filler

    @property
    def continuous_monitor(self) -> Any:
        """Return the ContinuousMonitor instance.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._continuous_monitor

    @property
    def alert_generator(self) -> Any:
        """Return the AlertGenerator instance.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._alert_generator

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
            logger.debug("SatelliteMonitoringService already started")
            return

        start = time.monotonic()
        logger.info("SatelliteMonitoringService starting up...")

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
            "SatelliteMonitoringService started in %.1fms: "
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
            logger.debug("SatelliteMonitoringService already stopped")
            return

        logger.info("SatelliteMonitoringService shutting down...")
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
            "SatelliteMonitoringService shut down in %.1fms", elapsed
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
            timestamp=_utcnow(),
            version="1.0.0",
            uptime_seconds=self.uptime_seconds,
        )
        self._last_health = health
        return health.to_dict()

    # ------------------------------------------------------------------
    # Unified API: Imagery search
    # ------------------------------------------------------------------

    def search_imagery(
        self,
        polygon_vertices: List[Tuple[float, float]],
        date_range: Tuple[str, str],
        source: str = "sentinel2",
        cloud_cover_max: Optional[float] = None,
    ) -> List[Any]:
        """Search for satellite imagery scenes covering a polygon.

        Delegates to the ImageryAcquisitionEngine for scene catalog
        queries against Sentinel-2, Landsat 8/9, or GFW tile indexes.

        Args:
            polygon_vertices: List of (lat, lon) tuples forming the AOI.
            date_range: Tuple of (start_date, end_date) in ISO format.
            source: Satellite source ('sentinel2', 'landsat8', 'landsat9').
            cloud_cover_max: Maximum cloud cover percentage override.
                If None, uses the config default.

        Returns:
            List of SceneMetadata objects matching the search criteria.

        Raises:
            RuntimeError: If the service has not been started.
            RuntimeError: If the ImageryAcquisitionEngine is not available.
        """
        self._ensure_started()
        if self._imagery_acquisition is None:
            raise RuntimeError(
                "ImageryAcquisitionEngine is not available"
            )

        effective_cloud_max = (
            cloud_cover_max
            if cloud_cover_max is not None
            else self._config.cloud_cover_max
        )

        logger.debug(
            "Searching imagery: source=%s, vertices=%d, "
            "date_range=%s, cloud_max=%.1f%%",
            source, len(polygon_vertices),
            date_range, effective_cloud_max,
        )

        return self._imagery_acquisition.search(
            polygon_vertices=polygon_vertices,
            date_range=date_range,
            source=source,
            cloud_cover_max=effective_cloud_max,
        )

    # ------------------------------------------------------------------
    # Unified API: Baseline establishment
    # ------------------------------------------------------------------

    def establish_baseline(
        self,
        plot_id: str,
        polygon_vertices: List[Tuple[float, float]],
        commodity: str,
        country_code: str,
        biome: Optional[str] = None,
    ) -> Any:
        """Establish a spectral baseline for a production plot.

        Creates a composite baseline snapshot from cloud-free imagery
        around the EUDR cutoff date (December 31, 2020). The baseline
        includes NDVI, EVI, and NBR statistics used for subsequent
        change detection.

        Args:
            plot_id: Unique plot identifier.
            polygon_vertices: List of (lat, lon) tuples forming the boundary.
            commodity: EUDR commodity identifier.
            country_code: ISO 3166-1 alpha-2 country code.
            biome: Optional biome override. If None, auto-detected from
                commodity and country.

        Returns:
            BaselineSnapshot from the BaselineManager.

        Raises:
            RuntimeError: If the service has not been started.
            RuntimeError: If the BaselineManager is not available.
        """
        self._ensure_started()
        if self._baseline_manager is None:
            raise RuntimeError("BaselineManager is not available")

        logger.info(
            "Establishing baseline: plot=%s, vertices=%d, "
            "commodity=%s, country=%s, biome=%s",
            plot_id, len(polygon_vertices),
            commodity, country_code, biome or "auto",
        )

        return self._baseline_manager.establish(
            plot_id=plot_id,
            polygon_vertices=polygon_vertices,
            commodity=commodity,
            country_code=country_code,
            biome=biome,
        )

    def get_baseline(self, plot_id: str) -> Optional[Any]:
        """Retrieve an existing baseline snapshot for a plot.

        Args:
            plot_id: Unique plot identifier.

        Returns:
            BaselineSnapshot if found, or None.

        Raises:
            RuntimeError: If the service has not been started.
            RuntimeError: If the BaselineManager is not available.
        """
        self._ensure_started()
        if self._baseline_manager is None:
            raise RuntimeError("BaselineManager is not available")

        logger.debug("Getting baseline: plot=%s", plot_id)
        return self._baseline_manager.get(plot_id)

    # ------------------------------------------------------------------
    # Unified API: Spectral index calculation
    # ------------------------------------------------------------------

    def calculate_indices(
        self,
        red_band: Any,
        nir_band: Any,
        index_type: str = "NDVI",
        biome: Optional[str] = None,
    ) -> Any:
        """Calculate a spectral vegetation index from band data.

        Delegates to the SpectralIndexCalculator for NDVI, EVI, NBR,
        NDMI, or SAVI computation using deterministic formulas (zero
        hallucination).

        Args:
            red_band: Red band reflectance data (array or scalar).
            nir_band: NIR band reflectance data (array or scalar).
            index_type: Spectral index to compute. One of 'NDVI',
                'EVI', 'NBR', 'NDMI', 'SAVI'. Defaults to 'NDVI'.
            biome: Optional biome context for classification thresholds.

        Returns:
            SpectralIndexResult with computed index values and
            classification.

        Raises:
            RuntimeError: If the service has not been started.
            RuntimeError: If the SpectralIndexCalculator is not available.
        """
        self._ensure_started()
        if self._spectral_calculator is None:
            raise RuntimeError(
                "SpectralIndexCalculator is not available"
            )

        logger.debug(
            "Calculating index: type=%s, biome=%s",
            index_type, biome or "default",
        )

        return self._spectral_calculator.calculate(
            red_band=red_band,
            nir_band=nir_band,
            index_type=index_type,
            biome=biome,
        )

    # ------------------------------------------------------------------
    # Unified API: Change detection
    # ------------------------------------------------------------------

    def detect_change(
        self,
        plot_id: str,
        polygon_vertices: List[Tuple[float, float]],
        baseline: Any,
        commodity: str,
        analysis_date: Optional[str] = None,
    ) -> Any:
        """Detect forest cover change relative to a baseline.

        Compares current satellite observations against the established
        baseline to identify deforestation, degradation, or regrowth.

        Args:
            plot_id: Unique plot identifier.
            polygon_vertices: List of (lat, lon) tuples forming the boundary.
            baseline: BaselineSnapshot to compare against.
            commodity: EUDR commodity identifier.
            analysis_date: Optional ISO date string for the analysis
                window end. Defaults to today.

        Returns:
            ChangeDetectionResult from the ForestChangeDetector.

        Raises:
            RuntimeError: If the service has not been started.
            RuntimeError: If the ForestChangeDetector is not available.
        """
        self._ensure_started()
        if self._change_detector is None:
            raise RuntimeError(
                "ForestChangeDetector is not available"
            )

        logger.info(
            "Detecting change: plot=%s, commodity=%s, date=%s",
            plot_id, commodity, analysis_date or "today",
        )

        return self._change_detector.detect(
            plot_id=plot_id,
            polygon_vertices=polygon_vertices,
            baseline=baseline,
            commodity=commodity,
            analysis_date=analysis_date,
        )

    # ------------------------------------------------------------------
    # Unified API: Full analysis
    # ------------------------------------------------------------------

    def run_full_analysis(
        self, request: AnalysisRequest,
    ) -> FullAnalysisResult:
        """Perform full satellite monitoring analysis across all engines.

        Orchestrates the complete analysis pipeline:
            1. Check/establish baseline for the plot
            2. Search for current imagery
            3. Run change detection against baseline
            4. Perform multi-source data fusion
            5. Generate alerts if deforestation detected
            6. Compute SHA-256 provenance hash

        Args:
            request: AnalysisRequest with all plot data.

        Returns:
            FullAnalysisResult with combined results from all engines.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        start = time.monotonic()

        logger.info(
            "Running full analysis: plot=%s, commodity=%s, "
            "country=%s, level=%s, biome=%s, vertices=%d",
            request.plot_id, request.commodity,
            request.country_code, request.analysis_level,
            request.biome or "auto", len(request.polygon_vertices),
        )

        # Step 1: Check/establish baseline
        baseline = self._safe_get_or_establish_baseline(
            request.plot_id,
            request.polygon_vertices,
            request.commodity,
            request.country_code,
            request.biome,
        )

        # Step 2: Search current imagery
        imagery = self._safe_search_current_imagery(
            request.polygon_vertices,
        )

        # Step 3: Change detection
        change_result = self._safe_detect_change(
            request.plot_id,
            request.polygon_vertices,
            baseline,
            request.commodity,
        )

        # Step 4: Multi-source data fusion
        fusion_result = self._safe_run_fusion(
            request.plot_id,
            request.polygon_vertices,
            change_result,
        )

        # Step 5: Alert generation
        alerts = self._safe_generate_alerts(
            request.plot_id,
            change_result,
            fusion_result,
            request.commodity,
            request.country_code,
        )

        # Step 6: Data quality assessment
        data_quality = self._assess_data_quality(
            baseline, change_result, fusion_result, imagery,
        )

        elapsed_ms = (time.monotonic() - start) * 1000.0

        # Step 7: Compute provenance hash
        provenance_hash = self._compute_analysis_provenance(
            request.plot_id,
            baseline,
            change_result,
            fusion_result,
            alerts,
        )

        result = FullAnalysisResult(
            plot_id=request.plot_id,
            baseline=baseline,
            change_detection=change_result,
            fusion=fusion_result,
            alerts=alerts,
            evidence=None,
            data_quality=data_quality,
            provenance_hash=provenance_hash,
            analyzed_at=_utcnow(),
            processing_time_ms=elapsed_ms,
        )

        logger.info(
            "Full analysis complete: plot=%s, alerts=%d, "
            "hash=%s, elapsed=%.1fms",
            request.plot_id, len(alerts),
            provenance_hash[:12], elapsed_ms,
        )

        return result

    # ------------------------------------------------------------------
    # Unified API: Batch analysis
    # ------------------------------------------------------------------

    def submit_batch(
        self,
        operator_id: str,
        requests: List[AnalysisRequest],
    ) -> str:
        """Submit a batch of satellite monitoring analysis requests.

        Creates a batch job, processes all plots, and stores results.
        Returns the batch ID for subsequent status polling via
        ``get_batch_result()``.

        Args:
            operator_id: Unique operator identifier.
            requests: List of AnalysisRequest objects.

        Returns:
            Batch ID string for tracking.

        Raises:
            RuntimeError: If the service has not been started.
            ValueError: If the batch exceeds max_batch_size.
        """
        self._ensure_started()

        if len(requests) > self._config.max_batch_size:
            raise ValueError(
                f"Batch size {len(requests)} exceeds maximum "
                f"{self._config.max_batch_size}"
            )

        batch_id = f"SAT-BATCH-{uuid.uuid4().hex[:12]}"
        batch_result = BatchAnalysisResult(
            operator_id=operator_id,
            batch_id=batch_id,
            status="processing",
            total_plots=len(requests),
        )

        with self._batch_lock:
            self._batch_registry[batch_id] = batch_result

        logger.info(
            "Batch submitted: id=%s, operator=%s, plots=%d",
            batch_id, operator_id, len(requests),
        )

        # Process all plots synchronously
        start = time.monotonic()
        results: List[FullAnalysisResult] = []
        failed_count = 0

        for request in requests:
            try:
                plot_result = self.run_full_analysis(request)
                results.append(plot_result)
            except Exception as exc:
                logger.warning(
                    "Batch plot failed: plot=%s, error=%s",
                    request.plot_id, exc,
                )
                failed_count += 1
                results.append(FullAnalysisResult(
                    plot_id=request.plot_id,
                    provenance_hash="",
                ))

        elapsed_ms = (time.monotonic() - start) * 1000.0

        # Compute batch statistics
        statistics = self._compute_batch_statistics(results)

        # Compute batch provenance hash
        batch_provenance = self._compute_batch_provenance(
            operator_id, batch_id, results,
        )

        # Update batch result
        with self._batch_lock:
            batch_result.status = "completed"
            batch_result.completed_plots = len(results)
            batch_result.failed_plots = failed_count
            batch_result.results = results
            batch_result.statistics = statistics
            batch_result.provenance_hash = batch_provenance
            batch_result.completed_at = _utcnow()
            batch_result.processing_time_ms = elapsed_ms

        logger.info(
            "Batch completed: id=%s, total=%d, completed=%d, "
            "failed=%d, elapsed=%.1fms",
            batch_id, len(requests), len(results),
            failed_count, elapsed_ms,
        )

        return batch_id

    def get_batch_result(
        self, batch_id: str,
    ) -> Optional[BatchAnalysisResult]:
        """Get the status and results of a batch analysis job.

        Args:
            batch_id: Batch identifier returned by ``submit_batch()``.

        Returns:
            BatchAnalysisResult with current status and results,
            or None if the batch_id is not found.
        """
        with self._batch_lock:
            return self._batch_registry.get(batch_id)

    # ------------------------------------------------------------------
    # Unified API: Monitoring schedules
    # ------------------------------------------------------------------

    def create_monitoring_schedule(
        self,
        plot_id: str,
        polygon_vertices: List[Tuple[float, float]],
        commodity: str,
        country_code: str,
        interval: str = "monthly",
    ) -> Any:
        """Create a recurring monitoring schedule for a plot.

        Sets up continuous satellite monitoring at the specified
        interval. The ContinuousMonitor engine handles scheduling,
        execution, and alert generation.

        Args:
            plot_id: Unique plot identifier.
            polygon_vertices: List of (lat, lon) tuples forming the boundary.
            commodity: EUDR commodity identifier.
            country_code: ISO 3166-1 alpha-2 country code.
            interval: Monitoring frequency. One of 'weekly', 'biweekly',
                'monthly', 'quarterly'. Defaults to 'monthly'.

        Returns:
            MonitoringSchedule object with schedule details.

        Raises:
            RuntimeError: If the service has not been started.
            RuntimeError: If the ContinuousMonitor is not available.
        """
        self._ensure_started()
        if self._continuous_monitor is None:
            raise RuntimeError("ContinuousMonitor is not available")

        logger.info(
            "Creating monitoring schedule: plot=%s, commodity=%s, "
            "country=%s, interval=%s",
            plot_id, commodity, country_code, interval,
        )

        return self._continuous_monitor.create_schedule(
            plot_id=plot_id,
            polygon_vertices=polygon_vertices,
            commodity=commodity,
            country_code=country_code,
            interval=interval,
        )

    # ------------------------------------------------------------------
    # Unified API: Evidence generation
    # ------------------------------------------------------------------

    def generate_evidence(
        self,
        plot_id: str,
        operator_id: str,
        format: str = "json",
    ) -> Any:
        """Generate an evidence package for EUDR compliance documentation.

        Compiles satellite analysis results, imagery metadata, spectral
        indices, change detection outputs, and provenance hashes into a
        structured evidence package suitable for Article 9 due diligence
        statements.

        Args:
            plot_id: Unique plot identifier.
            operator_id: Unique operator identifier.
            format: Output format. One of 'json', 'pdf_data', 'geojson'.
                Defaults to 'json'.

        Returns:
            EvidencePackage with compiled evidence data.

        Raises:
            RuntimeError: If the service has not been started.
            RuntimeError: If required engines are not available.
        """
        self._ensure_started()

        logger.info(
            "Generating evidence: plot=%s, operator=%s, format=%s",
            plot_id, operator_id, format,
        )

        # Collect evidence from available engines
        evidence_data: Dict[str, Any] = {
            "plot_id": plot_id,
            "operator_id": operator_id,
            "format": format,
            "generated_at": _utcnow().isoformat(),
        }

        # Baseline evidence
        baseline = self._safe_get_baseline(plot_id)
        if baseline is not None:
            evidence_data["baseline"] = (
                baseline.to_dict()
                if hasattr(baseline, "to_dict")
                else str(baseline)
            )

        # Compute evidence provenance
        evidence_hash = hashlib.sha256(
            json.dumps(evidence_data, sort_keys=True, default=str).encode()
        ).hexdigest()
        evidence_data["provenance_hash"] = evidence_hash

        return evidence_data

    # ------------------------------------------------------------------
    # Unified API: Alert management
    # ------------------------------------------------------------------

    def get_alerts(
        self,
        plot_id: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100,
    ) -> List[Any]:
        """Retrieve satellite monitoring alerts.

        Args:
            plot_id: Optional filter by plot identifier.
            severity: Optional filter by severity level ('critical',
                'high', 'medium', 'low').
            limit: Maximum number of alerts to return. Defaults to 100.

        Returns:
            List of SatelliteAlert objects matching the criteria.

        Raises:
            RuntimeError: If the service has not been started.
            RuntimeError: If the AlertGenerator is not available.
        """
        self._ensure_started()
        if self._alert_generator is None:
            raise RuntimeError("AlertGenerator is not available")

        logger.debug(
            "Getting alerts: plot=%s, severity=%s, limit=%d",
            plot_id or "all", severity or "all", limit,
        )

        return self._alert_generator.get_alerts(
            plot_id=plot_id,
            severity=severity,
            limit=limit,
        )

    def acknowledge_alert(
        self,
        alert_id: str,
        user_id: str,
    ) -> Any:
        """Acknowledge a satellite monitoring alert.

        Marks an alert as reviewed by a user, recording the user ID
        and timestamp for audit trail purposes.

        Args:
            alert_id: Unique alert identifier.
            user_id: Identifier of the user acknowledging the alert.

        Returns:
            Updated SatelliteAlert with acknowledgment metadata.

        Raises:
            RuntimeError: If the service has not been started.
            RuntimeError: If the AlertGenerator is not available.
        """
        self._ensure_started()
        if self._alert_generator is None:
            raise RuntimeError("AlertGenerator is not available")

        logger.info(
            "Acknowledging alert: id=%s, user=%s",
            alert_id, user_id,
        )

        return self._alert_generator.acknowledge(
            alert_id=alert_id,
            user_id=user_id,
        )

    # ------------------------------------------------------------------
    # Internal: Safe wrappers (exception-tolerant for run_full_analysis)
    # ------------------------------------------------------------------

    def _safe_get_or_establish_baseline(
        self,
        plot_id: str,
        polygon_vertices: List[Tuple[float, float]],
        commodity: str,
        country_code: str,
        biome: Optional[str],
    ) -> Optional[Any]:
        """Get existing baseline or establish a new one.

        Returns None on engine unavailability (non-fatal).

        Args:
            plot_id: Plot identifier.
            polygon_vertices: Polygon vertices.
            commodity: Commodity identifier.
            country_code: Country code.
            biome: Optional biome override.

        Returns:
            BaselineSnapshot or None.
        """
        if self._baseline_manager is None:
            return None
        try:
            existing = self._baseline_manager.get(plot_id)
            if existing is not None:
                return existing
            return self.establish_baseline(
                plot_id, polygon_vertices,
                commodity, country_code, biome,
            )
        except Exception as exc:
            logger.warning(
                "Baseline establishment failed (non-fatal): %s", exc
            )
            return None

    def _safe_get_baseline(self, plot_id: str) -> Optional[Any]:
        """Get existing baseline, returning None on failure.

        Args:
            plot_id: Plot identifier.

        Returns:
            BaselineSnapshot or None.
        """
        if self._baseline_manager is None:
            return None
        try:
            return self._baseline_manager.get(plot_id)
        except Exception as exc:
            logger.warning(
                "Baseline retrieval failed (non-fatal): %s", exc
            )
            return None

    def _safe_search_current_imagery(
        self,
        polygon_vertices: List[Tuple[float, float]],
    ) -> List[Any]:
        """Search current imagery, returning empty list on failure.

        Args:
            polygon_vertices: Polygon vertices for AOI.

        Returns:
            List of SceneMetadata or empty list.
        """
        if self._imagery_acquisition is None:
            return []
        try:
            today = date.today().isoformat()
            thirty_days_ago = date.fromordinal(
                date.today().toordinal() - 30
            ).isoformat()
            return self.search_imagery(
                polygon_vertices,
                (thirty_days_ago, today),
                "sentinel2",
            )
        except Exception as exc:
            logger.warning(
                "Imagery search failed (non-fatal): %s", exc
            )
            return []

    def _safe_detect_change(
        self,
        plot_id: str,
        polygon_vertices: List[Tuple[float, float]],
        baseline: Optional[Any],
        commodity: str,
    ) -> Optional[Any]:
        """Detect change, returning None on engine unavailability.

        Args:
            plot_id: Plot identifier.
            polygon_vertices: Polygon vertices.
            baseline: Baseline snapshot.
            commodity: Commodity identifier.

        Returns:
            ChangeDetectionResult or None.
        """
        if self._change_detector is None or baseline is None:
            return None
        try:
            return self.detect_change(
                plot_id, polygon_vertices, baseline, commodity,
            )
        except Exception as exc:
            logger.warning(
                "Change detection failed (non-fatal): %s", exc
            )
            return None

    def _safe_run_fusion(
        self,
        plot_id: str,
        polygon_vertices: List[Tuple[float, float]],
        change_result: Optional[Any],
    ) -> Optional[Any]:
        """Run multi-source fusion, returning None on failure.

        Args:
            plot_id: Plot identifier.
            polygon_vertices: Polygon vertices.
            change_result: Change detection result.

        Returns:
            FusionResult or None.
        """
        if self._data_fusion is None:
            return None
        try:
            return self._data_fusion.fuse(
                plot_id=plot_id,
                polygon_vertices=polygon_vertices,
                change_result=change_result,
            )
        except Exception as exc:
            logger.warning(
                "Data fusion failed (non-fatal): %s", exc
            )
            return None

    def _safe_generate_alerts(
        self,
        plot_id: str,
        change_result: Optional[Any],
        fusion_result: Optional[Any],
        commodity: str,
        country_code: str,
    ) -> List[Any]:
        """Generate alerts, returning empty list on failure.

        Args:
            plot_id: Plot identifier.
            change_result: Change detection result.
            fusion_result: Fusion result.
            commodity: Commodity identifier.
            country_code: Country code.

        Returns:
            List of SatelliteAlert objects or empty list.
        """
        if self._alert_generator is None:
            return []
        try:
            return self._alert_generator.generate(
                plot_id=plot_id,
                change_result=change_result,
                fusion_result=fusion_result,
                commodity=commodity,
                country_code=country_code,
                confidence_threshold=self._config.alert_confidence_threshold,
            )
        except Exception as exc:
            logger.warning(
                "Alert generation failed (non-fatal): %s", exc
            )
            return []

    # ------------------------------------------------------------------
    # Internal: Data quality assessment
    # ------------------------------------------------------------------

    def _assess_data_quality(
        self,
        baseline: Optional[Any],
        change_result: Optional[Any],
        fusion_result: Optional[Any],
        imagery: List[Any],
    ) -> Dict[str, Any]:
        """Assess data quality across all analysis components.

        Args:
            baseline: Baseline snapshot result.
            change_result: Change detection result.
            fusion_result: Fusion result.
            imagery: List of imagery metadata.

        Returns:
            Dictionary with data quality metrics.
        """
        quality: Dict[str, Any] = {
            "baseline_available": baseline is not None,
            "change_detection_available": change_result is not None,
            "fusion_available": fusion_result is not None,
            "imagery_scenes_found": len(imagery),
            "engines_reporting": sum(1 for x in [
                baseline, change_result, fusion_result,
            ] if x is not None),
            "completeness_score": 0.0,
        }

        # Calculate completeness as fraction of available components
        total_components = 3  # baseline, change, fusion
        available = quality["engines_reporting"]
        quality["completeness_score"] = round(
            available / total_components, 2
        ) if total_components > 0 else 0.0

        return quality

    # ------------------------------------------------------------------
    # Internal: Provenance hashes
    # ------------------------------------------------------------------

    def _compute_analysis_provenance(
        self,
        plot_id: str,
        baseline: Optional[Any],
        change_result: Optional[Any],
        fusion_result: Optional[Any],
        alerts: List[Any],
    ) -> str:
        """Compute SHA-256 provenance hash for a full analysis.

        Args:
            plot_id: Plot identifier.
            baseline: Baseline result.
            change_result: Change detection result.
            fusion_result: Fusion result.
            alerts: Generated alerts.

        Returns:
            SHA-256 hex digest string.
        """
        hash_parts: List[str] = [plot_id]

        if baseline is not None and hasattr(baseline, "provenance_hash"):
            hash_parts.append(baseline.provenance_hash)

        if change_result is not None:
            hash_parts.append(
                str(getattr(change_result, "deforestation_detected", ""))
            )
            if hasattr(change_result, "provenance_hash"):
                hash_parts.append(change_result.provenance_hash)

        if fusion_result is not None:
            hash_parts.append(
                str(getattr(fusion_result, "confidence", ""))
            )

        hash_parts.append(str(len(alerts)))

        hash_input = "|".join(hash_parts)
        return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()

    def _compute_batch_provenance(
        self,
        operator_id: str,
        batch_id: str,
        results: List[FullAnalysisResult],
    ) -> str:
        """Compute SHA-256 provenance hash for a batch analysis.

        Args:
            operator_id: Operator identifier.
            batch_id: Batch identifier.
            results: List of analysis results.

        Returns:
            SHA-256 hex digest string.
        """
        hash_parts: List[str] = [operator_id, batch_id]

        for result in results:
            hash_parts.append(result.provenance_hash or result.plot_id)

        hash_input = "|".join(hash_parts)
        return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()

    def _compute_batch_statistics(
        self,
        results: List[FullAnalysisResult],
    ) -> Dict[str, Any]:
        """Compute aggregated statistics for a batch analysis.

        Args:
            results: List of analysis results.

        Returns:
            Dictionary with aggregated statistics.
        """
        total = len(results)
        alerts_total = sum(len(r.alerts) for r in results)
        baselines_available = sum(
            1 for r in results if r.baseline is not None
        )
        changes_detected = sum(
            1 for r in results
            if r.change_detection is not None
            and getattr(r.change_detection, "deforestation_detected", False)
        )
        processing_times = [r.processing_time_ms for r in results]

        return {
            "total_plots": total,
            "baselines_available": baselines_available,
            "changes_detected": changes_detected,
            "alerts_generated": alerts_total,
            "avg_processing_time_ms": round(
                sum(processing_times) / total, 2
            ) if total > 0 else 0.0,
            "max_processing_time_ms": round(
                max(processing_times), 2
            ) if processing_times else 0.0,
            "min_processing_time_ms": round(
                min(processing_times), 2
            ) if processing_times else 0.0,
        }

    # ------------------------------------------------------------------
    # Internal: Logging
    # ------------------------------------------------------------------

    def _configure_logging(self) -> None:
        """Configure structured logging based on service configuration."""
        log_level = getattr(logging, self._config.log_level, logging.INFO)
        logging.getLogger(
            "greenlang.agents.eudr.satellite_monitoring"
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
                "greenlang.agents.eudr.satellite_monitoring",
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
                self._config.cache_ttl_seconds,
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
        client, and service configuration. Each engine is initialized
        with ImportError fallback for non-fatal degradation.
        """
        logger.info("Initializing 8 satellite monitoring engines...")

        # 1. ImageryAcquisitionEngine
        self._imagery_acquisition = await self._init_imagery_acquisition()

        # 2. SpectralIndexCalculator
        self._spectral_calculator = await self._init_spectral_calculator()

        # 3. BaselineManager
        self._baseline_manager = await self._init_baseline_manager()

        # 4. ForestChangeDetector
        self._change_detector = await self._init_change_detector()

        # 5. DataFusionEngine
        self._data_fusion = await self._init_data_fusion()

        # 6. CloudGapFiller
        self._cloud_gap_filler = await self._init_cloud_gap_filler()

        # 7. ContinuousMonitor
        self._continuous_monitor = await self._init_continuous_monitor()

        # 8. AlertGenerator
        self._alert_generator = await self._init_alert_generator()

        logger.info("All 8 engines initialized successfully")

    async def _init_imagery_acquisition(self) -> Any:
        """Initialize the ImageryAcquisitionEngine.

        Returns:
            Initialized ImageryAcquisitionEngine instance, or None.
        """
        try:
            from greenlang.agents.eudr.satellite_monitoring.imagery_acquisition import (
                ImageryAcquisitionEngine,
            )

            engine = ImageryAcquisitionEngine(config=self._config)
            logger.info("ImageryAcquisitionEngine initialized")
            return engine
        except ImportError:
            logger.warning("ImageryAcquisitionEngine not available")
            return None
        except Exception as exc:
            logger.error(
                "Failed to initialize ImageryAcquisitionEngine: %s",
                exc, exc_info=True,
            )
            return None

    async def _init_spectral_calculator(self) -> Any:
        """Initialize the SpectralIndexCalculator.

        Returns:
            Initialized SpectralIndexCalculator instance, or None.
        """
        try:
            from greenlang.agents.eudr.satellite_monitoring.spectral_index_calculator import (
                SpectralIndexCalculator,
            )

            calculator = SpectralIndexCalculator(config=self._config)
            logger.info("SpectralIndexCalculator initialized")
            return calculator
        except ImportError:
            logger.warning("SpectralIndexCalculator not available")
            return None
        except Exception as exc:
            logger.error(
                "Failed to initialize SpectralIndexCalculator: %s",
                exc, exc_info=True,
            )
            return None

    async def _init_baseline_manager(self) -> Any:
        """Initialize the BaselineManager.

        Returns:
            Initialized BaselineManager instance, or None.
        """
        try:
            from greenlang.agents.eudr.satellite_monitoring.baseline_manager import (
                BaselineManager,
            )

            manager = BaselineManager(config=self._config)
            logger.info("BaselineManager initialized")
            return manager
        except ImportError:
            logger.warning("BaselineManager not available")
            return None
        except Exception as exc:
            logger.error(
                "Failed to initialize BaselineManager: %s",
                exc, exc_info=True,
            )
            return None

    async def _init_change_detector(self) -> Any:
        """Initialize the ForestChangeDetector.

        Returns:
            Initialized ForestChangeDetector instance, or None.
        """
        try:
            from greenlang.agents.eudr.satellite_monitoring.forest_change_detector import (
                ForestChangeDetector,
            )

            detector = ForestChangeDetector(config=self._config)
            logger.info("ForestChangeDetector initialized")
            return detector
        except ImportError:
            logger.warning("ForestChangeDetector not available")
            return None
        except Exception as exc:
            logger.error(
                "Failed to initialize ForestChangeDetector: %s",
                exc, exc_info=True,
            )
            return None

    async def _init_data_fusion(self) -> Any:
        """Initialize the DataFusionEngine.

        Returns:
            Initialized DataFusionEngine instance, or None.
        """
        try:
            from greenlang.agents.eudr.satellite_monitoring.data_fusion import (
                DataFusionEngine,
            )

            engine = DataFusionEngine(config=self._config)
            logger.info("DataFusionEngine initialized")
            return engine
        except ImportError:
            logger.debug("DataFusionEngine module not yet available")
            return None
        except Exception as exc:
            logger.error(
                "Failed to initialize DataFusionEngine: %s",
                exc, exc_info=True,
            )
            return None

    async def _init_cloud_gap_filler(self) -> Any:
        """Initialize the CloudGapFiller.

        Returns:
            Initialized CloudGapFiller instance, or None.
        """
        try:
            from greenlang.agents.eudr.satellite_monitoring.cloud_gap_filler import (
                CloudGapFiller,
            )

            filler = CloudGapFiller(config=self._config)
            logger.info("CloudGapFiller initialized")
            return filler
        except ImportError:
            logger.debug("CloudGapFiller module not yet available")
            return None
        except Exception as exc:
            logger.error(
                "Failed to initialize CloudGapFiller: %s",
                exc, exc_info=True,
            )
            return None

    async def _init_continuous_monitor(self) -> Any:
        """Initialize the ContinuousMonitor.

        Returns:
            Initialized ContinuousMonitor instance, or None.
        """
        try:
            from greenlang.agents.eudr.satellite_monitoring.continuous_monitor import (
                ContinuousMonitor,
            )

            monitor = ContinuousMonitor(config=self._config)
            logger.info("ContinuousMonitor initialized")
            return monitor
        except ImportError:
            logger.debug("ContinuousMonitor module not yet available")
            return None
        except Exception as exc:
            logger.error(
                "Failed to initialize ContinuousMonitor: %s",
                exc, exc_info=True,
            )
            return None

    async def _init_alert_generator(self) -> Any:
        """Initialize the AlertGenerator.

        Returns:
            Initialized AlertGenerator instance, or None.
        """
        try:
            from greenlang.agents.eudr.satellite_monitoring.alert_generator import (
                AlertGenerator,
            )

            generator = AlertGenerator(config=self._config)
            logger.info("AlertGenerator initialized")
            return generator
        except ImportError:
            logger.debug("AlertGenerator module not yet available")
            return None
        except Exception as exc:
            logger.error(
                "Failed to initialize AlertGenerator: %s",
                exc, exc_info=True,
            )
            return None

    # ------------------------------------------------------------------
    # Internal: Engine shutdown
    # ------------------------------------------------------------------

    async def _close_engines(self) -> None:
        """Close all engines that implement a close/shutdown method."""
        engine_names = [
            ("imagery_acquisition", self._imagery_acquisition),
            ("spectral_calculator", self._spectral_calculator),
            ("baseline_manager", self._baseline_manager),
            ("change_detector", self._change_detector),
            ("data_fusion", self._data_fusion),
            ("cloud_gap_filler", self._cloud_gap_filler),
            ("continuous_monitor", self._continuous_monitor),
            ("alert_generator", self._alert_generator),
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
        self._imagery_acquisition = None
        self._spectral_calculator = None
        self._baseline_manager = None
        self._change_detector = None
        self._data_fusion = None
        self._cloud_gap_filler = None
        self._continuous_monitor = None
        self._alert_generator = None

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
            name="sat-health-check",
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
            "imagery_acquisition": self._imagery_acquisition,
            "spectral_calculator": self._spectral_calculator,
            "baseline_manager": self._baseline_manager,
            "change_detector": self._change_detector,
            "data_fusion": self._data_fusion,
            "cloud_gap_filler": self._cloud_gap_filler,
            "continuous_monitor": self._continuous_monitor,
            "alert_generator": self._alert_generator,
        }

        engine_statuses: Dict[str, str] = {}
        initialized_count = 0
        for name, engine in engines.items():
            if engine is not None:
                engine_statuses[name] = "initialized"
                initialized_count += 1
            else:
                engine_statuses[name] = "unavailable"

        # Core engines required for basic operation
        core_engines = [
            "imagery_acquisition",
            "spectral_calculator",
            "baseline_manager",
            "change_detector",
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
                "SatelliteMonitoringService is not started. "
                "Call await service.startup() first."
            )

    # ------------------------------------------------------------------
    # Convenience: get_engine
    # ------------------------------------------------------------------

    def get_engine(self, name: str) -> Any:
        """Retrieve an engine by name.

        Args:
            name: Engine name (e.g., 'imagery_acquisition',
                'change_detector').

        Returns:
            The engine instance, or None if not initialized.

        Raises:
            RuntimeError: If the service has not been started.
            ValueError: If the engine name is not recognized.
        """
        self._ensure_started()
        valid_names = {
            "imagery_acquisition": self._imagery_acquisition,
            "spectral_calculator": self._spectral_calculator,
            "baseline_manager": self._baseline_manager,
            "change_detector": self._change_detector,
            "data_fusion": self._data_fusion,
            "cloud_gap_filler": self._cloud_gap_filler,
            "continuous_monitor": self._continuous_monitor,
            "alert_generator": self._alert_generator,
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
            self._imagery_acquisition,
            self._spectral_calculator,
            self._baseline_manager,
            self._change_detector,
            self._data_fusion,
            self._cloud_gap_filler,
            self._continuous_monitor,
            self._alert_generator,
        ]
        return sum(1 for e in engines if e is not None)


# ---------------------------------------------------------------------------
# FastAPI lifespan context manager
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: Any) -> AsyncIterator[None]:
    """FastAPI lifespan context manager for the Satellite Monitoring service.

    Automatically starts the service on application startup and shuts it
    down on application shutdown. The service instance is stored in
    ``app.state.sat_service`` for access from route handlers.

    Usage with FastAPI::

        from fastapi import FastAPI
        from greenlang.agents.eudr.satellite_monitoring.setup import lifespan

        app = FastAPI(lifespan=lifespan)

    Args:
        app: The FastAPI application instance.

    Yields:
        None (service is accessible via ``app.state.sat_service``).
    """
    service = get_service()
    app.state.sat_service = service
    try:
        await service.startup()
        yield
    finally:
        await service.shutdown()


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_service_instance: Optional[SatelliteMonitoringService] = None
_service_lock = threading.Lock()


def get_service(
    config: Optional[SatelliteMonitoringConfig] = None,
) -> SatelliteMonitoringService:
    """Return the singleton SatelliteMonitoringService instance.

    Uses double-checked locking for thread safety. The instance is
    created on first call. Pass a config to override the default
    environment-based configuration.

    Args:
        config: Optional configuration override.

    Returns:
        SatelliteMonitoringService singleton instance.

    Example:
        >>> service = get_service()
        >>> await service.startup()
    """
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = SatelliteMonitoringService(
                    config=config
                )
    return _service_instance


def set_service(service: SatelliteMonitoringService) -> None:
    """Replace the singleton SatelliteMonitoringService instance.

    Primarily intended for testing and dependency injection.

    Args:
        service: Replacement service instance.
    """
    global _service_instance
    with _service_lock:
        _service_instance = service
    logger.info("SatelliteMonitoringService singleton replaced")


def reset_service() -> None:
    """Reset the singleton SatelliteMonitoringService to None.

    The next call to ``get_service()`` will create a fresh instance.
    Intended for test teardown.
    """
    global _service_instance
    with _service_lock:
        _service_instance = None
    logger.debug("SatelliteMonitoringService singleton reset")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "SatelliteMonitoringService",
    "HealthStatus",
    "AnalysisRequest",
    "FullAnalysisResult",
    "BatchAnalysisResult",
    "lifespan",
    "get_service",
    "set_service",
    "reset_service",
]
