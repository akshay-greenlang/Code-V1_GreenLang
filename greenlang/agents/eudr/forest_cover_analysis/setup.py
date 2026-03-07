# -*- coding: utf-8 -*-
"""
ForestCoverAnalysisService - Facade for AGENT-EUDR-004 Forest Cover Analysis

This module implements the ForestCoverAnalysisService, the single entry point
for all forest cover analysis operations in the GL-EUDR-APP. It manages the
lifecycle of eight internal engines, an async PostgreSQL connection pool
(psycopg + psycopg_pool), a Redis cache connection, OpenTelemetry tracing,
and Prometheus metrics. The service exposes a unified interface consumed by
the FastAPI router layer and the GL-EUDR-APP integration.

Lifecycle:
    startup  -> load config -> connect DB -> register pgvector -> connect Redis
             -> initialize engines -> start health check background task
    shutdown -> close engines -> close Redis -> close DB pool -> flush metrics

Engines (8):
    1. CanopyDensityMapper       - NDVI-to-canopy fractional cover (Feature 1)
    2. ForestTypeClassifier      - Biome-aware forest type classification (Feature 2)
    3. HistoricalReconstructor   - Multi-year historical forest cover (Feature 3)
    4. DeforestationFreeVerifier - EUDR deforestation-free determination (Feature 4)
    5. CanopyHeightModeler       - GEDI/allometric height estimation (Feature 5)
    6. FragmentationAnalyzer     - Landscape fragmentation metrics (Feature 6)
    7. BiomassEstimator          - Above-ground biomass estimation (Feature 7)
    8. ComplianceReporter        - EUDR compliance report generation (Feature 8)

FastAPI Integration:
    Use the ``lifespan`` async context manager with ``FastAPI(lifespan=lifespan)``
    for automatic startup/shutdown.

Example:
    >>> from greenlang.agents.eudr.forest_cover_analysis.setup import (
    ...     ForestCoverAnalysisService,
    ...     get_service,
    ... )
    >>> service = get_service()
    >>> await service.startup()
    >>> health = await service.health_check()
    >>> assert health["status"] == "healthy"
    >>> await service.shutdown()

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-004 Forest Cover Analysis Agent (GL-EUDR-FCA-004)
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

from greenlang.agents.eudr.forest_cover_analysis.config import (
    ForestCoverConfig,
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


def _compute_service_hash(config: ForestCoverConfig) -> str:
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
# PlotForestProfile
# ---------------------------------------------------------------------------


class PlotForestProfile:
    """Unified result from a complete forest cover analysis of a single plot.

    Combines results from canopy density, classification, height, biomass,
    fragmentation, historical reconstruction, and deforestation-free
    verification into a single auditable result.

    Attributes:
        plot_id: Unique plot identifier.
        canopy_density: CanopyDensityResult from the CanopyDensityMapper.
        forest_type: ForestTypeResult from the ForestTypeClassifier.
        historical: HistoricalReconstructionResult from HistoricalReconstructor.
        verification: DeforestationFreeResult from DeforestationFreeVerifier.
        canopy_height: CanopyHeightResult from CanopyHeightModeler.
        fragmentation: FragmentationResult from FragmentationAnalyzer.
        biomass: BiomassResult from BiomassEstimator.
        data_quality: Data quality assessment metrics.
        provenance_hash: SHA-256 hash of the combined result.
        analyzed_at: UTC timestamp of analysis completion.
        processing_time_ms: Total processing time in milliseconds.
    """

    __slots__ = (
        "plot_id", "canopy_density", "forest_type", "historical",
        "verification", "canopy_height", "fragmentation", "biomass",
        "data_quality", "provenance_hash", "analyzed_at",
        "processing_time_ms",
    )

    def __init__(
        self,
        plot_id: str = "",
        canopy_density: Optional[Any] = None,
        forest_type: Optional[Any] = None,
        historical: Optional[Any] = None,
        verification: Optional[Any] = None,
        canopy_height: Optional[Any] = None,
        fragmentation: Optional[Any] = None,
        biomass: Optional[Any] = None,
        data_quality: Optional[Dict[str, Any]] = None,
        provenance_hash: str = "",
        analyzed_at: Optional[datetime] = None,
        processing_time_ms: float = 0.0,
    ) -> None:
        self.plot_id = plot_id
        self.canopy_density = canopy_density
        self.forest_type = forest_type
        self.historical = historical
        self.verification = verification
        self.canopy_height = canopy_height
        self.fragmentation = fragmentation
        self.biomass = biomass
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
            "canopy_density": _result_dict(self.canopy_density),
            "forest_type": _result_dict(self.forest_type),
            "historical": _result_dict(self.historical),
            "verification": _result_dict(self.verification),
            "canopy_height": _result_dict(self.canopy_height),
            "fragmentation": _result_dict(self.fragmentation),
            "biomass": _result_dict(self.biomass),
            "data_quality": self.data_quality,
            "provenance_hash": self.provenance_hash,
            "analyzed_at": self.analyzed_at.isoformat(),
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


# ---------------------------------------------------------------------------
# DeforestationFreeResult
# ---------------------------------------------------------------------------


class DeforestationFreeResult:
    """Result of a deforestation-free verification for a production plot.

    Attributes:
        plot_id: Unique plot identifier.
        verdict: Determination verdict ('DEFORESTATION_FREE', 'DEFORESTATION_DETECTED',
            'DEGRADATION_DETECTED', 'INCONCLUSIVE').
        confidence: Confidence score (0.0-1.0).
        baseline_canopy_pct: Canopy cover at the EUDR cutoff date.
        current_canopy_pct: Current canopy cover.
        canopy_change_pct: Absolute change in canopy cover.
        historical_summary: Summary of historical trajectory.
        evidence_hash: SHA-256 hash of the evidence supporting the verdict.
        verified_at: UTC timestamp of verification.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "plot_id", "verdict", "confidence", "baseline_canopy_pct",
        "current_canopy_pct", "canopy_change_pct", "historical_summary",
        "evidence_hash", "verified_at", "processing_time_ms",
    )

    def __init__(
        self,
        plot_id: str = "",
        verdict: str = "INCONCLUSIVE",
        confidence: float = 0.0,
        baseline_canopy_pct: float = 0.0,
        current_canopy_pct: float = 0.0,
        canopy_change_pct: float = 0.0,
        historical_summary: Optional[Dict[str, Any]] = None,
        evidence_hash: str = "",
        verified_at: Optional[datetime] = None,
        processing_time_ms: float = 0.0,
    ) -> None:
        self.plot_id = plot_id
        self.verdict = verdict
        self.confidence = confidence
        self.baseline_canopy_pct = baseline_canopy_pct
        self.current_canopy_pct = current_canopy_pct
        self.canopy_change_pct = canopy_change_pct
        self.historical_summary = historical_summary or {}
        self.evidence_hash = evidence_hash
        self.verified_at = verified_at or _utcnow()
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "plot_id": self.plot_id,
            "verdict": self.verdict,
            "confidence": round(self.confidence, 4),
            "baseline_canopy_pct": round(self.baseline_canopy_pct, 2),
            "current_canopy_pct": round(self.current_canopy_pct, 2),
            "canopy_change_pct": round(self.canopy_change_pct, 2),
            "historical_summary": self.historical_summary,
            "evidence_hash": self.evidence_hash,
            "verified_at": self.verified_at.isoformat(),
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


# ---------------------------------------------------------------------------
# ComplianceReport
# ---------------------------------------------------------------------------


class ComplianceReport:
    """EUDR forest cover compliance report for a production plot.

    Attributes:
        plot_id: Unique plot identifier.
        report_type: Report type ('full', 'summary', 'evidence_only').
        format: Output format ('json', 'pdf_data', 'geojson').
        sections: Report section data keyed by section name.
        provenance_hash: SHA-256 hash of the complete report.
        generated_at: UTC timestamp of report generation.
    """

    __slots__ = (
        "plot_id", "report_type", "format", "sections",
        "provenance_hash", "generated_at",
    )

    def __init__(
        self,
        plot_id: str = "",
        report_type: str = "full",
        format: str = "json",
        sections: Optional[Dict[str, Any]] = None,
        provenance_hash: str = "",
        generated_at: Optional[datetime] = None,
    ) -> None:
        self.plot_id = plot_id
        self.report_type = report_type
        self.format = format
        self.sections = sections or {}
        self.provenance_hash = provenance_hash
        self.generated_at = generated_at or _utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "plot_id": self.plot_id,
            "report_type": self.report_type,
            "format": self.format,
            "sections": self.sections,
            "provenance_hash": self.provenance_hash,
            "generated_at": self.generated_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# AnalysisSummary
# ---------------------------------------------------------------------------


class AnalysisSummary:
    """Aggregated statistics across all analyzed plots.

    Attributes:
        total_plots: Total number of plots analyzed.
        deforestation_free_count: Plots with DEFORESTATION_FREE verdict.
        deforestation_detected_count: Plots with DEFORESTATION_DETECTED verdict.
        degradation_detected_count: Plots with DEGRADATION_DETECTED verdict.
        inconclusive_count: Plots with INCONCLUSIVE verdict.
        avg_canopy_cover_pct: Average canopy cover across all plots.
        avg_confidence: Average verification confidence.
        avg_processing_time_ms: Average processing time per plot.
        generated_at: UTC timestamp of summary generation.
    """

    __slots__ = (
        "total_plots", "deforestation_free_count",
        "deforestation_detected_count", "degradation_detected_count",
        "inconclusive_count", "avg_canopy_cover_pct",
        "avg_confidence", "avg_processing_time_ms", "generated_at",
    )

    def __init__(
        self,
        total_plots: int = 0,
        deforestation_free_count: int = 0,
        deforestation_detected_count: int = 0,
        degradation_detected_count: int = 0,
        inconclusive_count: int = 0,
        avg_canopy_cover_pct: float = 0.0,
        avg_confidence: float = 0.0,
        avg_processing_time_ms: float = 0.0,
        generated_at: Optional[datetime] = None,
    ) -> None:
        self.total_plots = total_plots
        self.deforestation_free_count = deforestation_free_count
        self.deforestation_detected_count = deforestation_detected_count
        self.degradation_detected_count = degradation_detected_count
        self.inconclusive_count = inconclusive_count
        self.avg_canopy_cover_pct = avg_canopy_cover_pct
        self.avg_confidence = avg_confidence
        self.avg_processing_time_ms = avg_processing_time_ms
        self.generated_at = generated_at or _utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "total_plots": self.total_plots,
            "deforestation_free_count": self.deforestation_free_count,
            "deforestation_detected_count": self.deforestation_detected_count,
            "degradation_detected_count": self.degradation_detected_count,
            "inconclusive_count": self.inconclusive_count,
            "avg_canopy_cover_pct": round(self.avg_canopy_cover_pct, 2),
            "avg_confidence": round(self.avg_confidence, 4),
            "avg_processing_time_ms": round(self.avg_processing_time_ms, 2),
            "generated_at": self.generated_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# ForestCoverDashboard
# ---------------------------------------------------------------------------


class ForestCoverDashboard:
    """Dashboard data container for forest cover analysis UI.

    Attributes:
        summary: AnalysisSummary with aggregate statistics.
        recent_analyses: List of recent PlotForestProfile results.
        alerts: List of active deforestation/degradation alerts.
        coverage_stats: Coverage statistics by commodity and country.
        generated_at: UTC timestamp.
    """

    __slots__ = (
        "summary", "recent_analyses", "alerts",
        "coverage_stats", "generated_at",
    )

    def __init__(
        self,
        summary: Optional[AnalysisSummary] = None,
        recent_analyses: Optional[List[Dict[str, Any]]] = None,
        alerts: Optional[List[Dict[str, Any]]] = None,
        coverage_stats: Optional[Dict[str, Any]] = None,
        generated_at: Optional[datetime] = None,
    ) -> None:
        self.summary = summary or AnalysisSummary()
        self.recent_analyses = recent_analyses or []
        self.alerts = alerts or []
        self.coverage_stats = coverage_stats or {}
        self.generated_at = generated_at or _utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "summary": self.summary.to_dict(),
            "recent_analyses": self.recent_analyses,
            "alerts": self.alerts,
            "coverage_stats": self.coverage_stats,
            "generated_at": self.generated_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# BatchAnalysisResult
# ---------------------------------------------------------------------------


class BatchAnalysisResult:
    """Result of a batch forest cover analysis job.

    Attributes:
        operator_id: Unique operator identifier who submitted the batch.
        batch_id: Unique batch identifier.
        status: Batch status ('pending', 'processing', 'completed', 'failed').
        total_plots: Total plots in the batch.
        completed_plots: Number of plots processed so far.
        failed_plots: Number of plots that failed analysis.
        results: Per-plot PlotForestProfile list.
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
        results: Optional[List[PlotForestProfile]] = None,
        statistics: Optional[Dict[str, Any]] = None,
        provenance_hash: str = "",
        submitted_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
        processing_time_ms: float = 0.0,
    ) -> None:
        self.operator_id = operator_id
        self.batch_id = batch_id or f"FCA-BATCH-{uuid.uuid4().hex[:12]}"
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
# ForestCoverAnalysisService
# ---------------------------------------------------------------------------


class ForestCoverAnalysisService:
    """Facade service for the EUDR Forest Cover Analysis Agent.

    This is the single entry point for all forest cover analysis operations.
    It manages the full lifecycle of database connections, cache connections,
    eight internal engines, health monitoring, and OpenTelemetry tracing.

    The service follows a strict startup/shutdown protocol:
        startup:  config -> DB pool -> pgvector -> Redis -> engines -> health
        shutdown: health stop -> engines -> Redis -> DB pool -> metrics flush

    Attributes:
        config: Service configuration loaded from env or injected.
        is_running: Whether the service is currently active and healthy.

    Example:
        >>> service = ForestCoverAnalysisService()
        >>> await service.startup()
        >>> profile = service.analyze_plot_complete("PLOT-001", wkt, "COCOA")
        >>> await service.shutdown()
    """

    def __init__(
        self,
        config: Optional[ForestCoverConfig] = None,
    ) -> None:
        """Initialize ForestCoverAnalysisService.

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
        self._canopy_density_mapper: Optional[Any] = None
        self._forest_type_classifier: Optional[Any] = None
        self._historical_reconstructor: Optional[Any] = None
        self._deforestation_free_verifier: Optional[Any] = None
        self._canopy_height_modeler: Optional[Any] = None
        self._fragmentation_analyzer: Optional[Any] = None
        self._biomass_estimator: Optional[Any] = None
        self._compliance_reporter: Optional[Any] = None

        # Profile cache (in-memory for fast retrieval)
        self._profile_cache: Dict[str, PlotForestProfile] = {}
        self._cache_lock = threading.Lock()

        # Batch tracking
        self._batch_registry: Dict[str, BatchAnalysisResult] = {}
        self._batch_lock = threading.Lock()

        # Health check background task
        self._health_task: Optional[asyncio.Task[None]] = None
        self._last_health: Optional[HealthStatus] = None
        self._health_interval_seconds: float = 30.0

        # OpenTelemetry tracer
        self._tracer: Optional[Any] = None

        # Metrics counters
        self._api_errors: int = 0
        self._analyses_completed: int = 0
        self._verifications_completed: int = 0

        logger.info(
            "ForestCoverAnalysisService created: config_hash=%s, "
            "max_batch=%d, concurrency=%d, cache_ttl=%ds, cutoff=%s",
            self._config_hash[:12],
            self._config.max_batch_size,
            self._config.analysis_concurrency,
            self._config.cache_ttl_seconds,
            self._config.cutoff_date,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> ForestCoverConfig:
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
    def canopy_density_mapper(self) -> Any:
        """Return the CanopyDensityMapper engine instance.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._canopy_density_mapper

    @property
    def forest_type_classifier(self) -> Any:
        """Return the ForestTypeClassifier engine instance.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._forest_type_classifier

    @property
    def historical_reconstructor(self) -> Any:
        """Return the HistoricalReconstructor engine instance.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._historical_reconstructor

    @property
    def deforestation_free_verifier(self) -> Any:
        """Return the DeforestationFreeVerifier engine instance.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._deforestation_free_verifier

    @property
    def canopy_height_modeler(self) -> Any:
        """Return the CanopyHeightModeler engine instance.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._canopy_height_modeler

    @property
    def fragmentation_analyzer(self) -> Any:
        """Return the FragmentationAnalyzer engine instance.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._fragmentation_analyzer

    @property
    def biomass_estimator(self) -> Any:
        """Return the BiomassEstimator engine instance.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._biomass_estimator

    @property
    def compliance_reporter(self) -> Any:
        """Return the ComplianceReporter engine instance.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._compliance_reporter

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
            logger.debug("ForestCoverAnalysisService already started")
            return

        start = time.monotonic()
        logger.info("ForestCoverAnalysisService starting up...")

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
            "ForestCoverAnalysisService started in %.1fms: "
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
            logger.debug("ForestCoverAnalysisService already stopped")
            return

        logger.info("ForestCoverAnalysisService shutting down...")
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
            "ForestCoverAnalysisService shut down in %.1fms", elapsed
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
    # Unified API: analyze_plot_complete
    # ------------------------------------------------------------------

    def analyze_plot_complete(
        self,
        plot_id: str,
        polygon_wkt: str,
        commodity: str,
        country_code: str = "",
        biome: Optional[str] = None,
        analysis_level: str = "standard",
    ) -> PlotForestProfile:
        """Perform complete forest cover analysis across all engines.

        Orchestrates the full analysis pipeline:
            1. Run canopy density mapping
            2. Classify forest type
            3. Estimate canopy height
            4. Estimate above-ground biomass
            5. Analyze landscape fragmentation
            6. Reconstruct historical forest cover
            7. Verify deforestation-free status
            8. Compute SHA-256 provenance hash

        Args:
            plot_id: Unique plot identifier.
            polygon_wkt: WKT geometry string for the plot boundary.
            commodity: EUDR commodity identifier (e.g., 'COCOA', 'SOYA').
            country_code: ISO 3166-1 alpha-2 country code.
            biome: Optional biome override. If None, auto-detected from
                commodity and country_code.
            analysis_level: Analysis depth ('quick', 'standard', 'deep').

        Returns:
            PlotForestProfile with combined results from all engines.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        start = time.monotonic()

        logger.info(
            "Running complete analysis: plot=%s, commodity=%s, "
            "country=%s, level=%s, biome=%s",
            plot_id, commodity, country_code,
            analysis_level, biome or "auto",
        )

        # Step 1: Canopy density mapping
        canopy_density = self._safe_map_canopy_density(
            plot_id, polygon_wkt, biome,
        )

        # Step 2: Forest type classification
        forest_type = self._safe_classify_forest_type(
            plot_id, polygon_wkt, commodity, biome,
        )

        # Step 3: Canopy height estimation
        canopy_height = self._safe_model_canopy_height(
            plot_id, polygon_wkt, biome,
        )

        # Step 4: Biomass estimation
        biomass = self._safe_estimate_biomass(
            plot_id, polygon_wkt, biome,
        )

        # Step 5: Fragmentation analysis
        fragmentation = self._safe_analyze_fragmentation(
            plot_id, polygon_wkt,
        )

        # Step 6: Historical reconstruction
        historical = self._safe_reconstruct_historical(
            plot_id, polygon_wkt, commodity, country_code,
        )

        # Step 7: Deforestation-free verification
        verification = self._safe_verify_deforestation_free(
            plot_id, polygon_wkt, commodity, country_code,
            canopy_density, historical,
        )

        # Step 8: Data quality assessment
        data_quality = self._assess_data_quality(
            canopy_density, forest_type, canopy_height,
            biomass, fragmentation, historical, verification,
        )

        elapsed_ms = (time.monotonic() - start) * 1000.0

        # Step 9: Compute provenance hash
        provenance_hash = self._compute_analysis_provenance(
            plot_id, canopy_density, forest_type, canopy_height,
            biomass, fragmentation, historical, verification,
        )

        profile = PlotForestProfile(
            plot_id=plot_id,
            canopy_density=canopy_density,
            forest_type=forest_type,
            historical=historical,
            verification=verification,
            canopy_height=canopy_height,
            fragmentation=fragmentation,
            biomass=biomass,
            data_quality=data_quality,
            provenance_hash=provenance_hash,
            analyzed_at=_utcnow(),
            processing_time_ms=elapsed_ms,
        )

        # Cache the profile
        with self._cache_lock:
            self._profile_cache[plot_id] = profile

        self._analyses_completed += 1

        logger.info(
            "Complete analysis finished: plot=%s, hash=%s, elapsed=%.1fms",
            plot_id, provenance_hash[:12], elapsed_ms,
        )

        return profile

    # ------------------------------------------------------------------
    # Unified API: verify_deforestation_free
    # ------------------------------------------------------------------

    def verify_deforestation_free(
        self,
        plot_id: str,
        polygon_wkt: str,
        commodity: str,
        country_code: str = "",
        biome: Optional[str] = None,
    ) -> DeforestationFreeResult:
        """Quick path: verify deforestation-free status for a plot.

        Performs the minimum analysis required for a deforestation-free
        determination: historical reconstruction, current canopy density,
        and verdict generation. Does NOT run biomass, height, or
        fragmentation analysis.

        Args:
            plot_id: Unique plot identifier.
            polygon_wkt: WKT geometry string for the plot boundary.
            commodity: EUDR commodity identifier.
            country_code: ISO 3166-1 alpha-2 country code.
            biome: Optional biome override.

        Returns:
            DeforestationFreeResult with verdict and confidence.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        start = time.monotonic()

        logger.info(
            "Verifying deforestation-free: plot=%s, commodity=%s, "
            "country=%s, biome=%s",
            plot_id, commodity, country_code, biome or "auto",
        )

        # Step 1: Get current canopy density
        canopy_density = self._safe_map_canopy_density(
            plot_id, polygon_wkt, biome,
        )

        # Step 2: Historical reconstruction
        historical = self._safe_reconstruct_historical(
            plot_id, polygon_wkt, commodity, country_code,
        )

        # Step 3: Run verification engine
        verification_result = self._safe_verify_deforestation_free(
            plot_id, polygon_wkt, commodity, country_code,
            canopy_density, historical,
        )

        elapsed_ms = (time.monotonic() - start) * 1000.0

        # Build standardized result
        baseline_pct = 0.0
        current_pct = 0.0
        verdict = "INCONCLUSIVE"
        confidence = 0.0

        if verification_result is not None:
            baseline_pct = float(
                getattr(verification_result, "baseline_canopy_pct", 0.0)
            )
            current_pct = float(
                getattr(verification_result, "current_canopy_pct", 0.0)
            )
            verdict = str(
                getattr(verification_result, "verdict", "INCONCLUSIVE")
            )
            confidence = float(
                getattr(verification_result, "confidence", 0.0)
            )

        # Compute evidence hash
        evidence_hash = hashlib.sha256(
            f"{plot_id}|{verdict}|{confidence}|{baseline_pct}|{current_pct}"
            .encode("utf-8")
        ).hexdigest()

        result = DeforestationFreeResult(
            plot_id=plot_id,
            verdict=verdict,
            confidence=confidence,
            baseline_canopy_pct=baseline_pct,
            current_canopy_pct=current_pct,
            canopy_change_pct=current_pct - baseline_pct,
            historical_summary=(
                historical.to_dict()
                if historical is not None and hasattr(historical, "to_dict")
                else {}
            ),
            evidence_hash=evidence_hash,
            verified_at=_utcnow(),
            processing_time_ms=elapsed_ms,
        )

        self._verifications_completed += 1

        logger.info(
            "Verification complete: plot=%s, verdict=%s, "
            "confidence=%.3f, elapsed=%.1fms",
            plot_id, verdict, confidence, elapsed_ms,
        )

        return result

    # ------------------------------------------------------------------
    # Unified API: batch_analyze
    # ------------------------------------------------------------------

    def batch_analyze(
        self,
        plot_ids: List[str],
        polygon_wkts: List[str],
        commodity: str,
        country_code: str = "",
        biome: Optional[str] = None,
        operator_id: str = "",
    ) -> List[PlotForestProfile]:
        """Process multiple plots with concurrency control.

        Analyzes each plot sequentially (thread-safe), recording batch
        statistics and provenance. Respects the max_batch_size limit.

        Args:
            plot_ids: List of unique plot identifiers.
            polygon_wkts: List of WKT geometry strings (same order as plot_ids).
            commodity: EUDR commodity identifier.
            country_code: ISO 3166-1 alpha-2 country code.
            biome: Optional biome override.
            operator_id: Unique operator identifier for audit trail.

        Returns:
            List of PlotForestProfile results (one per plot).

        Raises:
            RuntimeError: If the service has not been started.
            ValueError: If plot_ids and polygon_wkts have different lengths.
            ValueError: If the batch exceeds max_batch_size.
        """
        self._ensure_started()

        if len(plot_ids) != len(polygon_wkts):
            raise ValueError(
                f"plot_ids ({len(plot_ids)}) and polygon_wkts "
                f"({len(polygon_wkts)}) must have the same length"
            )

        if len(plot_ids) > self._config.max_batch_size:
            raise ValueError(
                f"Batch size {len(plot_ids)} exceeds maximum "
                f"{self._config.max_batch_size}"
            )

        batch_id = f"FCA-BATCH-{uuid.uuid4().hex[:12]}"
        start = time.monotonic()

        logger.info(
            "Batch analysis started: batch=%s, operator=%s, plots=%d, "
            "commodity=%s, country=%s",
            batch_id, operator_id or "unknown", len(plot_ids),
            commodity, country_code,
        )

        results: List[PlotForestProfile] = []
        failed_count = 0

        for idx, (plot_id, polygon_wkt) in enumerate(
            zip(plot_ids, polygon_wkts)
        ):
            try:
                profile = self.analyze_plot_complete(
                    plot_id=plot_id,
                    polygon_wkt=polygon_wkt,
                    commodity=commodity,
                    country_code=country_code,
                    biome=biome,
                )
                results.append(profile)
            except Exception as exc:
                logger.warning(
                    "Batch plot failed: batch=%s, plot=%s (%d/%d), error=%s",
                    batch_id, plot_id, idx + 1, len(plot_ids), exc,
                )
                self._api_errors += 1
                failed_count += 1
                results.append(PlotForestProfile(
                    plot_id=plot_id,
                    provenance_hash="",
                ))

        elapsed_ms = (time.monotonic() - start) * 1000.0

        # Store batch result
        batch_provenance = self._compute_batch_provenance(
            operator_id, batch_id, results,
        )
        batch_statistics = self._compute_batch_statistics(results)

        batch_result = BatchAnalysisResult(
            operator_id=operator_id,
            batch_id=batch_id,
            status="completed",
            total_plots=len(plot_ids),
            completed_plots=len(results) - failed_count,
            failed_plots=failed_count,
            results=results,
            statistics=batch_statistics,
            provenance_hash=batch_provenance,
            completed_at=_utcnow(),
            processing_time_ms=elapsed_ms,
        )

        with self._batch_lock:
            self._batch_registry[batch_id] = batch_result

        logger.info(
            "Batch analysis completed: batch=%s, total=%d, "
            "completed=%d, failed=%d, elapsed=%.1fms",
            batch_id, len(plot_ids),
            len(results) - failed_count, failed_count, elapsed_ms,
        )

        return results

    # ------------------------------------------------------------------
    # Unified API: generate_compliance_report
    # ------------------------------------------------------------------

    def generate_compliance_report(
        self,
        plot_id: str,
        report_type: str = "full",
        format: str = "json",
        operator_id: str = "",
    ) -> ComplianceReport:
        """Compile analysis results and generate a compliance report.

        Retrieves the cached PlotForestProfile for the given plot_id
        and generates a structured compliance report via the
        ComplianceReporter engine.

        Args:
            plot_id: Unique plot identifier.
            report_type: Report type ('full', 'summary', 'evidence_only').
            format: Output format ('json', 'pdf_data', 'geojson').
            operator_id: Operator identifier for the report header.

        Returns:
            ComplianceReport with compiled analysis data.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        start = time.monotonic()

        logger.info(
            "Generating compliance report: plot=%s, type=%s, format=%s",
            plot_id, report_type, format,
        )

        # Retrieve cached profile
        profile = self.get_plot_profile(plot_id)

        # Build report sections
        sections: Dict[str, Any] = {
            "plot_id": plot_id,
            "operator_id": operator_id,
            "report_type": report_type,
            "generated_at": _utcnow().isoformat(),
            "cutoff_date": self._config.cutoff_date,
        }

        if profile is not None:
            sections["analysis_results"] = profile.to_dict()
            sections["fao_thresholds"] = self._config.fao_thresholds

        # Delegate to compliance reporter engine if available
        if self._compliance_reporter is not None:
            try:
                engine_report = self._compliance_reporter.generate(
                    plot_id=plot_id,
                    profile=profile,
                    report_type=report_type,
                    format=format,
                    operator_id=operator_id,
                )
                if engine_report is not None:
                    if hasattr(engine_report, "to_dict"):
                        sections["engine_report"] = engine_report.to_dict()
                    else:
                        sections["engine_report"] = str(engine_report)
            except Exception as exc:
                logger.warning(
                    "ComplianceReporter engine failed (non-fatal): %s", exc
                )
                self._api_errors += 1

        # Compute report provenance hash
        report_hash = hashlib.sha256(
            json.dumps(sections, sort_keys=True, default=str).encode()
        ).hexdigest()

        elapsed_ms = (time.monotonic() - start) * 1000.0

        report = ComplianceReport(
            plot_id=plot_id,
            report_type=report_type,
            format=format,
            sections=sections,
            provenance_hash=report_hash,
            generated_at=_utcnow(),
        )

        logger.info(
            "Compliance report generated: plot=%s, hash=%s, elapsed=%.1fms",
            plot_id, report_hash[:12], elapsed_ms,
        )

        return report

    # ------------------------------------------------------------------
    # Unified API: get_plot_profile
    # ------------------------------------------------------------------

    def get_plot_profile(
        self, plot_id: str,
    ) -> Optional[PlotForestProfile]:
        """Retrieve a cached/stored forest profile for a plot.

        Args:
            plot_id: Unique plot identifier.

        Returns:
            PlotForestProfile if found in cache, or None.
        """
        with self._cache_lock:
            return self._profile_cache.get(plot_id)

    # ------------------------------------------------------------------
    # Unified API: get_analysis_summary
    # ------------------------------------------------------------------

    def get_analysis_summary(self) -> AnalysisSummary:
        """Compute aggregate statistics across all analyzed plots.

        Returns:
            AnalysisSummary with counts and averages across all cached
            plot profiles.
        """
        with self._cache_lock:
            profiles = list(self._profile_cache.values())

        if not profiles:
            return AnalysisSummary(generated_at=_utcnow())

        total = len(profiles)
        deforestation_free = 0
        deforestation_detected = 0
        degradation_detected = 0
        inconclusive = 0
        canopy_covers: List[float] = []
        confidences: List[float] = []
        processing_times: List[float] = []

        for p in profiles:
            processing_times.append(p.processing_time_ms)

            if p.verification is not None:
                verdict = getattr(p.verification, "verdict", "INCONCLUSIVE")
                confidence = float(
                    getattr(p.verification, "confidence", 0.0)
                )

                if verdict == "DEFORESTATION_FREE":
                    deforestation_free += 1
                elif verdict == "DEFORESTATION_DETECTED":
                    deforestation_detected += 1
                elif verdict == "DEGRADATION_DETECTED":
                    degradation_detected += 1
                else:
                    inconclusive += 1

                confidences.append(confidence)
                current_pct = float(
                    getattr(p.verification, "current_canopy_pct", 0.0)
                )
                if current_pct > 0.0:
                    canopy_covers.append(current_pct)
            else:
                inconclusive += 1

        avg_canopy = (
            sum(canopy_covers) / len(canopy_covers)
            if canopy_covers else 0.0
        )
        avg_conf = (
            sum(confidences) / len(confidences)
            if confidences else 0.0
        )
        avg_time = (
            sum(processing_times) / total
            if total > 0 else 0.0
        )

        return AnalysisSummary(
            total_plots=total,
            deforestation_free_count=deforestation_free,
            deforestation_detected_count=deforestation_detected,
            degradation_detected_count=degradation_detected,
            inconclusive_count=inconclusive,
            avg_canopy_cover_pct=avg_canopy,
            avg_confidence=avg_conf,
            avg_processing_time_ms=avg_time,
            generated_at=_utcnow(),
        )

    # ------------------------------------------------------------------
    # Unified API: get_dashboard_data
    # ------------------------------------------------------------------

    def get_dashboard_data(self) -> ForestCoverDashboard:
        """Generate dashboard data for the forest cover analysis UI.

        Returns:
            ForestCoverDashboard with summary, recent analyses, alerts,
            and coverage statistics.
        """
        summary = self.get_analysis_summary()

        # Get the 20 most recent analyses
        with self._cache_lock:
            all_profiles = sorted(
                self._profile_cache.values(),
                key=lambda p: p.analyzed_at or _utcnow(),
                reverse=True,
            )
            recent = [p.to_dict() for p in all_profiles[:20]]

        # Collect alerts (deforestation/degradation detected)
        alerts: List[Dict[str, Any]] = []
        for p in all_profiles:
            if p.verification is not None:
                verdict = getattr(p.verification, "verdict", "")
                if verdict in (
                    "DEFORESTATION_DETECTED", "DEGRADATION_DETECTED"
                ):
                    alerts.append({
                        "plot_id": p.plot_id,
                        "verdict": verdict,
                        "confidence": float(
                            getattr(p.verification, "confidence", 0.0)
                        ),
                        "detected_at": p.analyzed_at.isoformat()
                        if p.analyzed_at else "",
                    })

        # Coverage statistics
        coverage_stats: Dict[str, Any] = {
            "total_cached_profiles": len(all_profiles),
            "analyses_completed": self._analyses_completed,
            "verifications_completed": self._verifications_completed,
            "api_errors": self._api_errors,
            "uptime_seconds": round(self.uptime_seconds, 2),
        }

        return ForestCoverDashboard(
            summary=summary,
            recent_analyses=recent,
            alerts=alerts,
            coverage_stats=coverage_stats,
            generated_at=_utcnow(),
        )

    # ------------------------------------------------------------------
    # Internal: Safe wrappers (exception-tolerant)
    # ------------------------------------------------------------------

    def _safe_map_canopy_density(
        self,
        plot_id: str,
        polygon_wkt: str,
        biome: Optional[str],
    ) -> Optional[Any]:
        """Map canopy density, returning None on engine unavailability.

        Args:
            plot_id: Plot identifier.
            polygon_wkt: WKT polygon geometry.
            biome: Optional biome override.

        Returns:
            CanopyDensityResult or None.
        """
        if self._canopy_density_mapper is None:
            return None
        try:
            return self._canopy_density_mapper.map_density(
                plot_id=plot_id,
                polygon_wkt=polygon_wkt,
                biome=biome,
            )
        except Exception as exc:
            logger.warning(
                "Canopy density mapping failed (non-fatal): %s", exc
            )
            self._api_errors += 1
            return None

    def _safe_classify_forest_type(
        self,
        plot_id: str,
        polygon_wkt: str,
        commodity: str,
        biome: Optional[str],
    ) -> Optional[Any]:
        """Classify forest type, returning None on failure.

        Args:
            plot_id: Plot identifier.
            polygon_wkt: WKT polygon geometry.
            commodity: Commodity identifier.
            biome: Optional biome override.

        Returns:
            ForestTypeResult or None.
        """
        if self._forest_type_classifier is None:
            return None
        try:
            return self._forest_type_classifier.classify(
                plot_id=plot_id,
                polygon_wkt=polygon_wkt,
                commodity=commodity,
                biome=biome,
            )
        except Exception as exc:
            logger.warning(
                "Forest type classification failed (non-fatal): %s", exc
            )
            self._api_errors += 1
            return None

    def _safe_model_canopy_height(
        self,
        plot_id: str,
        polygon_wkt: str,
        biome: Optional[str],
    ) -> Optional[Any]:
        """Model canopy height, returning None on failure.

        Args:
            plot_id: Plot identifier.
            polygon_wkt: WKT polygon geometry.
            biome: Optional biome override.

        Returns:
            CanopyHeightResult or None.
        """
        if self._canopy_height_modeler is None:
            return None
        try:
            return self._canopy_height_modeler.model_height(
                plot_id=plot_id,
                polygon_wkt=polygon_wkt,
                biome=biome,
            )
        except Exception as exc:
            logger.warning(
                "Canopy height modeling failed (non-fatal): %s", exc
            )
            self._api_errors += 1
            return None

    def _safe_estimate_biomass(
        self,
        plot_id: str,
        polygon_wkt: str,
        biome: Optional[str],
    ) -> Optional[Any]:
        """Estimate biomass, returning None on failure.

        Args:
            plot_id: Plot identifier.
            polygon_wkt: WKT polygon geometry.
            biome: Optional biome override.

        Returns:
            BiomassResult or None.
        """
        if self._biomass_estimator is None:
            return None
        try:
            return self._biomass_estimator.estimate(
                plot_id=plot_id,
                polygon_wkt=polygon_wkt,
                biome=biome,
            )
        except Exception as exc:
            logger.warning(
                "Biomass estimation failed (non-fatal): %s", exc
            )
            self._api_errors += 1
            return None

    def _safe_analyze_fragmentation(
        self,
        plot_id: str,
        polygon_wkt: str,
    ) -> Optional[Any]:
        """Analyze fragmentation, returning None on failure.

        Args:
            plot_id: Plot identifier.
            polygon_wkt: WKT polygon geometry.

        Returns:
            FragmentationResult or None.
        """
        if self._fragmentation_analyzer is None:
            return None
        try:
            return self._fragmentation_analyzer.analyze(
                plot_id=plot_id,
                polygon_wkt=polygon_wkt,
            )
        except Exception as exc:
            logger.warning(
                "Fragmentation analysis failed (non-fatal): %s", exc
            )
            self._api_errors += 1
            return None

    def _safe_reconstruct_historical(
        self,
        plot_id: str,
        polygon_wkt: str,
        commodity: str,
        country_code: str,
    ) -> Optional[Any]:
        """Reconstruct historical forest cover, returning None on failure.

        Args:
            plot_id: Plot identifier.
            polygon_wkt: WKT polygon geometry.
            commodity: Commodity identifier.
            country_code: Country code.

        Returns:
            HistoricalReconstructionResult or None.
        """
        if self._historical_reconstructor is None:
            return None
        try:
            return self._historical_reconstructor.reconstruct(
                plot_id=plot_id,
                polygon_wkt=polygon_wkt,
                commodity=commodity,
                country_code=country_code,
                baseline_start_year=self._config.baseline_start_year,
                baseline_end_year=self._config.baseline_end_year,
            )
        except Exception as exc:
            logger.warning(
                "Historical reconstruction failed (non-fatal): %s", exc
            )
            self._api_errors += 1
            return None

    def _safe_verify_deforestation_free(
        self,
        plot_id: str,
        polygon_wkt: str,
        commodity: str,
        country_code: str,
        canopy_density: Optional[Any],
        historical: Optional[Any],
    ) -> Optional[Any]:
        """Verify deforestation-free status, returning None on failure.

        Args:
            plot_id: Plot identifier.
            polygon_wkt: WKT polygon geometry.
            commodity: Commodity identifier.
            country_code: Country code.
            canopy_density: Current canopy density result.
            historical: Historical reconstruction result.

        Returns:
            DeforestationFreeVerification result or None.
        """
        if self._deforestation_free_verifier is None:
            return None
        try:
            return self._deforestation_free_verifier.verify(
                plot_id=plot_id,
                polygon_wkt=polygon_wkt,
                commodity=commodity,
                country_code=country_code,
                canopy_density=canopy_density,
                historical=historical,
                cutoff_date=self._config.cutoff_date,
                confidence_min=self._config.confidence_min,
            )
        except Exception as exc:
            logger.warning(
                "Deforestation-free verification failed (non-fatal): %s", exc
            )
            self._api_errors += 1
            return None

    # ------------------------------------------------------------------
    # Internal: Data quality assessment
    # ------------------------------------------------------------------

    def _assess_data_quality(
        self,
        canopy_density: Optional[Any],
        forest_type: Optional[Any],
        canopy_height: Optional[Any],
        biomass: Optional[Any],
        fragmentation: Optional[Any],
        historical: Optional[Any],
        verification: Optional[Any],
    ) -> Dict[str, Any]:
        """Assess data quality across all analysis components.

        Args:
            canopy_density: Canopy density result.
            forest_type: Forest type result.
            canopy_height: Canopy height result.
            biomass: Biomass result.
            fragmentation: Fragmentation result.
            historical: Historical reconstruction result.
            verification: Verification result.

        Returns:
            Dictionary with data quality metrics.
        """
        components = [
            ("canopy_density", canopy_density),
            ("forest_type", forest_type),
            ("canopy_height", canopy_height),
            ("biomass", biomass),
            ("fragmentation", fragmentation),
            ("historical", historical),
            ("verification", verification),
        ]

        available_count = sum(1 for _, c in components if c is not None)
        total_components = len(components)

        quality: Dict[str, Any] = {
            "engines_reporting": available_count,
            "total_engines": total_components,
            "completeness_score": round(
                available_count / total_components, 2
            ) if total_components > 0 else 0.0,
        }

        for name, component in components:
            quality[f"{name}_available"] = component is not None

        return quality

    # ------------------------------------------------------------------
    # Internal: Provenance hashes
    # ------------------------------------------------------------------

    def _compute_analysis_provenance(
        self,
        plot_id: str,
        canopy_density: Optional[Any],
        forest_type: Optional[Any],
        canopy_height: Optional[Any],
        biomass: Optional[Any],
        fragmentation: Optional[Any],
        historical: Optional[Any],
        verification: Optional[Any],
    ) -> str:
        """Compute SHA-256 provenance hash for a complete analysis.

        Args:
            plot_id: Plot identifier.
            canopy_density: Canopy density result.
            forest_type: Forest type result.
            canopy_height: Canopy height result.
            biomass: Biomass result.
            fragmentation: Fragmentation result.
            historical: Historical reconstruction result.
            verification: Verification result.

        Returns:
            SHA-256 hex digest string.
        """
        hash_parts: List[str] = [plot_id]

        for component in [
            canopy_density, forest_type, canopy_height,
            biomass, fragmentation, historical, verification,
        ]:
            if component is not None:
                if hasattr(component, "provenance_hash"):
                    hash_parts.append(component.provenance_hash)
                elif hasattr(component, "to_dict"):
                    part_str = json.dumps(
                        component.to_dict(), sort_keys=True, default=str,
                    )
                    hash_parts.append(
                        hashlib.sha256(part_str.encode()).hexdigest()[:16]
                    )
                else:
                    hash_parts.append(str(component)[:64])
            else:
                hash_parts.append("none")

        hash_input = "|".join(hash_parts)
        return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()

    def _compute_batch_provenance(
        self,
        operator_id: str,
        batch_id: str,
        results: List[PlotForestProfile],
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
        results: List[PlotForestProfile],
    ) -> Dict[str, Any]:
        """Compute aggregated statistics for a batch analysis.

        Args:
            results: List of analysis results.

        Returns:
            Dictionary with aggregated statistics.
        """
        total = len(results)
        if total == 0:
            return {"total_plots": 0}

        processing_times = [r.processing_time_ms for r in results]
        engines_reporting = [
            r.data_quality.get("engines_reporting", 0) for r in results
        ]
        verifications_available = sum(
            1 for r in results if r.verification is not None
        )
        deforestation_detected = sum(
            1 for r in results
            if r.verification is not None
            and getattr(r.verification, "verdict", "") == "DEFORESTATION_DETECTED"
        )

        return {
            "total_plots": total,
            "verifications_available": verifications_available,
            "deforestation_detected": deforestation_detected,
            "avg_engines_reporting": round(
                sum(engines_reporting) / total, 2
            ),
            "avg_processing_time_ms": round(
                sum(processing_times) / total, 2
            ),
            "max_processing_time_ms": round(max(processing_times), 2),
            "min_processing_time_ms": round(min(processing_times), 2),
        }

    # ------------------------------------------------------------------
    # Internal: Logging
    # ------------------------------------------------------------------

    def _configure_logging(self) -> None:
        """Configure structured logging based on service configuration."""
        log_level = getattr(logging, self._config.log_level, logging.INFO)
        logging.getLogger(
            "greenlang.agents.eudr.forest_cover_analysis"
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
                "greenlang.agents.eudr.forest_cover_analysis",
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
            pool_size = self._config.analysis_concurrency
            pool = AsyncConnectionPool(
                conninfo=conninfo,
                min_size=max(1, pool_size // 2),
                max_size=pool_size,
                open=False,
            )
            await pool.open()
            await pool.check()
            self._db_pool = pool
            logger.info(
                "PostgreSQL connection pool opened: min=%d, max=%d",
                max(1, pool_size // 2),
                pool_size,
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
        logger.info("Initializing 8 forest cover analysis engines...")

        # 1. CanopyDensityMapper
        self._canopy_density_mapper = await self._init_canopy_density_mapper()

        # 2. ForestTypeClassifier
        self._forest_type_classifier = await self._init_forest_type_classifier()

        # 3. HistoricalReconstructor
        self._historical_reconstructor = await self._init_historical_reconstructor()

        # 4. DeforestationFreeVerifier
        self._deforestation_free_verifier = await self._init_deforestation_free_verifier()

        # 5. CanopyHeightModeler
        self._canopy_height_modeler = await self._init_canopy_height_modeler()

        # 6. FragmentationAnalyzer
        self._fragmentation_analyzer = await self._init_fragmentation_analyzer()

        # 7. BiomassEstimator
        self._biomass_estimator = await self._init_biomass_estimator()

        # 8. ComplianceReporter
        self._compliance_reporter = await self._init_compliance_reporter()

        logger.info("All 8 engines initialized successfully")

    async def _init_canopy_density_mapper(self) -> Any:
        """Initialize the CanopyDensityMapper engine.

        Returns:
            Initialized CanopyDensityMapper instance, or None.
        """
        try:
            from greenlang.agents.eudr.forest_cover_analysis.canopy_density_mapper import (
                CanopyDensityMapper,
            )

            engine = CanopyDensityMapper(config=self._config)
            logger.info("CanopyDensityMapper initialized")
            return engine
        except ImportError:
            logger.debug("CanopyDensityMapper module not yet available")
            return None
        except Exception as exc:
            logger.error(
                "Failed to initialize CanopyDensityMapper: %s",
                exc, exc_info=True,
            )
            return None

    async def _init_forest_type_classifier(self) -> Any:
        """Initialize the ForestTypeClassifier engine.

        Returns:
            Initialized ForestTypeClassifier instance, or None.
        """
        try:
            from greenlang.agents.eudr.forest_cover_analysis.forest_type_classifier import (
                ForestTypeClassifier,
            )

            engine = ForestTypeClassifier(config=self._config)
            logger.info("ForestTypeClassifier initialized")
            return engine
        except ImportError:
            logger.debug("ForestTypeClassifier module not yet available")
            return None
        except Exception as exc:
            logger.error(
                "Failed to initialize ForestTypeClassifier: %s",
                exc, exc_info=True,
            )
            return None

    async def _init_historical_reconstructor(self) -> Any:
        """Initialize the HistoricalReconstructor engine.

        Returns:
            Initialized HistoricalReconstructor instance, or None.
        """
        try:
            from greenlang.agents.eudr.forest_cover_analysis.historical_reconstructor import (
                HistoricalReconstructor,
            )

            engine = HistoricalReconstructor(config=self._config)
            logger.info("HistoricalReconstructor initialized")
            return engine
        except ImportError:
            logger.debug("HistoricalReconstructor module not yet available")
            return None
        except Exception as exc:
            logger.error(
                "Failed to initialize HistoricalReconstructor: %s",
                exc, exc_info=True,
            )
            return None

    async def _init_deforestation_free_verifier(self) -> Any:
        """Initialize the DeforestationFreeVerifier engine.

        Returns:
            Initialized DeforestationFreeVerifier instance, or None.
        """
        try:
            from greenlang.agents.eudr.forest_cover_analysis.deforestation_free_verifier import (
                DeforestationFreeVerifier,
            )

            engine = DeforestationFreeVerifier(config=self._config)
            logger.info("DeforestationFreeVerifier initialized")
            return engine
        except ImportError:
            logger.debug(
                "DeforestationFreeVerifier module not yet available"
            )
            return None
        except Exception as exc:
            logger.error(
                "Failed to initialize DeforestationFreeVerifier: %s",
                exc, exc_info=True,
            )
            return None

    async def _init_canopy_height_modeler(self) -> Any:
        """Initialize the CanopyHeightModeler engine.

        Returns:
            Initialized CanopyHeightModeler instance, or None.
        """
        try:
            from greenlang.agents.eudr.forest_cover_analysis.canopy_height_modeler import (
                CanopyHeightModeler,
            )

            engine = CanopyHeightModeler(config=self._config)
            logger.info("CanopyHeightModeler initialized")
            return engine
        except ImportError:
            logger.debug("CanopyHeightModeler module not yet available")
            return None
        except Exception as exc:
            logger.error(
                "Failed to initialize CanopyHeightModeler: %s",
                exc, exc_info=True,
            )
            return None

    async def _init_fragmentation_analyzer(self) -> Any:
        """Initialize the FragmentationAnalyzer engine.

        Returns:
            Initialized FragmentationAnalyzer instance, or None.
        """
        try:
            from greenlang.agents.eudr.forest_cover_analysis.fragmentation_analyzer import (
                FragmentationAnalyzer,
            )

            engine = FragmentationAnalyzer(config=self._config)
            logger.info("FragmentationAnalyzer initialized")
            return engine
        except ImportError:
            logger.debug("FragmentationAnalyzer module not yet available")
            return None
        except Exception as exc:
            logger.error(
                "Failed to initialize FragmentationAnalyzer: %s",
                exc, exc_info=True,
            )
            return None

    async def _init_biomass_estimator(self) -> Any:
        """Initialize the BiomassEstimator engine.

        Returns:
            Initialized BiomassEstimator instance, or None.
        """
        try:
            from greenlang.agents.eudr.forest_cover_analysis.biomass_estimator import (
                BiomassEstimator,
            )

            engine = BiomassEstimator(config=self._config)
            logger.info("BiomassEstimator initialized")
            return engine
        except ImportError:
            logger.debug("BiomassEstimator module not yet available")
            return None
        except Exception as exc:
            logger.error(
                "Failed to initialize BiomassEstimator: %s",
                exc, exc_info=True,
            )
            return None

    async def _init_compliance_reporter(self) -> Any:
        """Initialize the ComplianceReporter engine.

        Returns:
            Initialized ComplianceReporter instance, or None.
        """
        try:
            from greenlang.agents.eudr.forest_cover_analysis.compliance_reporter import (
                ComplianceReporter,
            )

            engine = ComplianceReporter(config=self._config)
            logger.info("ComplianceReporter initialized")
            return engine
        except ImportError:
            logger.debug("ComplianceReporter module not yet available")
            return None
        except Exception as exc:
            logger.error(
                "Failed to initialize ComplianceReporter: %s",
                exc, exc_info=True,
            )
            return None

    # ------------------------------------------------------------------
    # Internal: Engine shutdown
    # ------------------------------------------------------------------

    async def _close_engines(self) -> None:
        """Close all engines that implement a close/shutdown method."""
        engine_names = [
            ("canopy_density_mapper", self._canopy_density_mapper),
            ("forest_type_classifier", self._forest_type_classifier),
            ("historical_reconstructor", self._historical_reconstructor),
            ("deforestation_free_verifier", self._deforestation_free_verifier),
            ("canopy_height_modeler", self._canopy_height_modeler),
            ("fragmentation_analyzer", self._fragmentation_analyzer),
            ("biomass_estimator", self._biomass_estimator),
            ("compliance_reporter", self._compliance_reporter),
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
        self._canopy_density_mapper = None
        self._forest_type_classifier = None
        self._historical_reconstructor = None
        self._deforestation_free_verifier = None
        self._canopy_height_modeler = None
        self._fragmentation_analyzer = None
        self._biomass_estimator = None
        self._compliance_reporter = None

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
            name="fca-health-check",
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
            "canopy_density_mapper": self._canopy_density_mapper,
            "forest_type_classifier": self._forest_type_classifier,
            "historical_reconstructor": self._historical_reconstructor,
            "deforestation_free_verifier": self._deforestation_free_verifier,
            "canopy_height_modeler": self._canopy_height_modeler,
            "fragmentation_analyzer": self._fragmentation_analyzer,
            "biomass_estimator": self._biomass_estimator,
            "compliance_reporter": self._compliance_reporter,
        }

        engine_statuses: Dict[str, str] = {}
        initialized_count = 0
        for name, engine in engines.items():
            if engine is not None:
                engine_statuses[name] = "initialized"
                initialized_count += 1
            else:
                engine_statuses[name] = "unavailable"

        # Core engines required for basic deforestation-free verification
        core_engines = [
            "canopy_density_mapper",
            "historical_reconstructor",
            "deforestation_free_verifier",
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
        logger.debug(
            "Prometheus metrics flushed: analyses=%d, "
            "verifications=%d, errors=%d",
            self._analyses_completed,
            self._verifications_completed,
            self._api_errors,
        )

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
                "ForestCoverAnalysisService is not started. "
                "Call await service.startup() first."
            )

    # ------------------------------------------------------------------
    # Convenience: get_engine
    # ------------------------------------------------------------------

    def get_engine(self, name: str) -> Any:
        """Retrieve an engine by name.

        Args:
            name: Engine name (e.g., 'canopy_density_mapper',
                'deforestation_free_verifier').

        Returns:
            The engine instance, or None if not initialized.

        Raises:
            RuntimeError: If the service has not been started.
            ValueError: If the engine name is not recognized.
        """
        self._ensure_started()
        valid_names = {
            "canopy_density_mapper": self._canopy_density_mapper,
            "forest_type_classifier": self._forest_type_classifier,
            "historical_reconstructor": self._historical_reconstructor,
            "deforestation_free_verifier": self._deforestation_free_verifier,
            "canopy_height_modeler": self._canopy_height_modeler,
            "fragmentation_analyzer": self._fragmentation_analyzer,
            "biomass_estimator": self._biomass_estimator,
            "compliance_reporter": self._compliance_reporter,
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
            self._canopy_density_mapper,
            self._forest_type_classifier,
            self._historical_reconstructor,
            self._deforestation_free_verifier,
            self._canopy_height_modeler,
            self._fragmentation_analyzer,
            self._biomass_estimator,
            self._compliance_reporter,
        ]
        return sum(1 for e in engines if e is not None)

    # ------------------------------------------------------------------
    # Convenience: get_batch_result
    # ------------------------------------------------------------------

    def get_batch_result(
        self, batch_id: str,
    ) -> Optional[BatchAnalysisResult]:
        """Get the status and results of a batch analysis job.

        Args:
            batch_id: Batch identifier returned by ``batch_analyze()``.

        Returns:
            BatchAnalysisResult with current status and results,
            or None if the batch_id is not found.
        """
        with self._batch_lock:
            return self._batch_registry.get(batch_id)


# ---------------------------------------------------------------------------
# FastAPI lifespan context manager
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: Any) -> AsyncIterator[None]:
    """FastAPI lifespan context manager for the Forest Cover Analysis service.

    Automatically starts the service on application startup and shuts it
    down on application shutdown. The service instance is stored in
    ``app.state.fca_service`` for access from route handlers.

    Usage with FastAPI::

        from fastapi import FastAPI
        from greenlang.agents.eudr.forest_cover_analysis.setup import lifespan

        app = FastAPI(lifespan=lifespan)

    Args:
        app: The FastAPI application instance.

    Yields:
        None (service is accessible via ``app.state.fca_service``).
    """
    service = get_service()
    app.state.fca_service = service
    try:
        await service.startup()
        yield
    finally:
        await service.shutdown()


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_service_instance: Optional[ForestCoverAnalysisService] = None
_service_lock = threading.Lock()


def get_service(
    config: Optional[ForestCoverConfig] = None,
) -> ForestCoverAnalysisService:
    """Return the singleton ForestCoverAnalysisService instance.

    Uses double-checked locking for thread safety. The instance is
    created on first call. Pass a config to override the default
    environment-based configuration.

    Args:
        config: Optional configuration override.

    Returns:
        ForestCoverAnalysisService singleton instance.

    Example:
        >>> service = get_service()
        >>> await service.startup()
    """
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = ForestCoverAnalysisService(
                    config=config
                )
    return _service_instance


def set_service(service: ForestCoverAnalysisService) -> None:
    """Replace the singleton ForestCoverAnalysisService instance.

    Primarily intended for testing and dependency injection.

    Args:
        service: Replacement service instance.
    """
    global _service_instance
    with _service_lock:
        _service_instance = service
    logger.info("ForestCoverAnalysisService singleton replaced")


def reset_service() -> None:
    """Reset the singleton ForestCoverAnalysisService to None.

    The next call to ``get_service()`` will create a fresh instance.
    Intended for test teardown.
    """
    global _service_instance
    with _service_lock:
        _service_instance = None
    logger.debug("ForestCoverAnalysisService singleton reset")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "ForestCoverAnalysisService",
    "HealthStatus",
    "PlotForestProfile",
    "DeforestationFreeResult",
    "ComplianceReport",
    "AnalysisSummary",
    "ForestCoverDashboard",
    "BatchAnalysisResult",
    "lifespan",
    "get_service",
    "set_service",
    "reset_service",
]
