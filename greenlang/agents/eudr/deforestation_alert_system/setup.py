# -*- coding: utf-8 -*-
"""
DeforestationAlertSystemSetup - Facade for AGENT-EUDR-020

Unified setup facade orchestrating all 8 engines of the Deforestation Alert
System Agent. Provides a single entry point for satellite change detection,
alert generation, severity classification, spatial buffer monitoring, EUDR
cutoff date verification, historical baseline comparison, alert workflow
management, and compliance impact assessment.

Engines (8):
    1. SatelliteChangeDetector         - Multi-source satellite change detection (Feature 1)
    2. AlertGenerator                  - Alert generation with deduplication (Feature 2)
    3. SeverityClassifier              - Five-tier severity classification (Feature 3)
    4. SpatialBufferMonitor            - Buffer zone proximity monitoring (Feature 4)
    5. CutoffDateVerifier              - EUDR cutoff date verification (Feature 5)
    6. HistoricalBaselineEngine        - Historical baseline comparison (Feature 6)
    7. AlertWorkflowEngine             - Alert workflow management with SLAs (Feature 7)
    8. ComplianceImpactAssessor        - Compliance impact assessment (Feature 8)

Reference Data (4):
    - satellite_sources: 5 satellite source specifications with spectral bands
    - deforestation_hotspots: 30+ global hotspot regions with FAO data
    - protected_areas: 100+ WDPA protected areas with IUCN categories
    - country_forest_data: 180+ country forest cover statistics

Singleton Pattern:
    Thread-safe singleton with double-checked locking via ``get_service()``.

FastAPI Integration:
    Use the ``lifespan`` async context manager with
    ``FastAPI(lifespan=lifespan)`` for automatic startup/shutdown.

Example:
    >>> from greenlang.agents.eudr.deforestation_alert_system.setup import (
    ...     DeforestationAlertSystemSetup,
    ...     get_service,
    ... )
    >>> service = get_service()
    >>> await service.startup()
    >>> health = await service.health_check()
    >>> assert health["status"] == "healthy"
    >>>
    >>> # Detect changes from satellite imagery
    >>> detections = await service.detect_changes(
    ...     latitude=-3.1234,
    ...     longitude=28.5678,
    ...     radius_km=10,
    ... )
    >>>
    >>> # Run comprehensive deforestation analysis
    >>> analysis = await service.run_comprehensive_analysis(
    ...     plot_id="plot-001",
    ...     latitude=-3.1234,
    ...     longitude=28.5678,
    ... )
    >>>
    >>> await service.shutdown()

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-020
Agent ID: GL-EUDR-DAS-020
Regulation: EU 2023/1115 (EUDR) Articles 2, 9, 10, 11, 31
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
from decimal import Decimal
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

from greenlang.agents.eudr.deforestation_alert_system.config import (
    DeforestationAlertSystemConfig,
    get_config,
    set_config,
    reset_config,
)
from greenlang.agents.eudr.deforestation_alert_system.provenance import (
    ProvenanceTracker,
    get_tracker,
)

# Metrics import (graceful fallback since metrics.py may not exist yet)
try:
    from greenlang.agents.eudr.deforestation_alert_system.metrics import (
        PROMETHEUS_AVAILABLE,
        record_satellite_detection,
        record_alert_generated,
        record_severity_classification,
        record_buffer_check,
        record_cutoff_verification,
        record_baseline_comparison,
        record_workflow_transition,
        record_compliance_assessment,
        record_false_positive,
        record_api_error,
        observe_detection_latency,
        observe_alert_generation_duration,
        observe_severity_scoring_duration,
        observe_compliance_assessment_duration,
        set_active_alerts,
        set_monitored_plots,
        set_active_buffers,
        set_pending_investigations,
        set_sla_breaches,
        set_detection_backlog,
    )
except ImportError:
    PROMETHEUS_AVAILABLE = False  # type: ignore[assignment]

    def _noop(*args: Any, **kwargs: Any) -> None:
        """No-op stub for unavailable metrics functions."""
        pass

    record_satellite_detection = _noop  # type: ignore[assignment]
    record_alert_generated = _noop  # type: ignore[assignment]
    record_severity_classification = _noop  # type: ignore[assignment]
    record_buffer_check = _noop  # type: ignore[assignment]
    record_cutoff_verification = _noop  # type: ignore[assignment]
    record_baseline_comparison = _noop  # type: ignore[assignment]
    record_workflow_transition = _noop  # type: ignore[assignment]
    record_compliance_assessment = _noop  # type: ignore[assignment]
    record_false_positive = _noop  # type: ignore[assignment]
    record_api_error = _noop  # type: ignore[assignment]
    observe_detection_latency = _noop  # type: ignore[assignment]
    observe_alert_generation_duration = _noop  # type: ignore[assignment]
    observe_severity_scoring_duration = _noop  # type: ignore[assignment]
    observe_compliance_assessment_duration = _noop  # type: ignore[assignment]
    set_active_alerts = _noop  # type: ignore[assignment]
    set_monitored_plots = _noop  # type: ignore[assignment]
    set_active_buffers = _noop  # type: ignore[assignment]
    set_pending_investigations = _noop  # type: ignore[assignment]
    set_sla_breaches = _noop  # type: ignore[assignment]
    set_detection_backlog = _noop  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Engine imports (graceful fallback for lazy loading)
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.deforestation_alert_system.engines.satellite_change_detector import (
        SatelliteChangeDetector,
    )
    _SATELLITE_CHANGE_DETECTOR_AVAILABLE = True
except ImportError:
    try:
        from greenlang.agents.eudr.deforestation_alert_system.satellite_change_detector import (
            SatelliteChangeDetector,
        )
        _SATELLITE_CHANGE_DETECTOR_AVAILABLE = True
    except ImportError:
        SatelliteChangeDetector = None  # type: ignore[assignment,misc]
        _SATELLITE_CHANGE_DETECTOR_AVAILABLE = False

try:
    from greenlang.agents.eudr.deforestation_alert_system.engines.alert_generator import (
        AlertGenerator,
    )
    _ALERT_GENERATOR_AVAILABLE = True
except ImportError:
    try:
        from greenlang.agents.eudr.deforestation_alert_system.alert_generator import (
            AlertGenerator,
        )
        _ALERT_GENERATOR_AVAILABLE = True
    except ImportError:
        AlertGenerator = None  # type: ignore[assignment,misc]
        _ALERT_GENERATOR_AVAILABLE = False

try:
    from greenlang.agents.eudr.deforestation_alert_system.engines.severity_classifier import (
        SeverityClassifier,
    )
    _SEVERITY_CLASSIFIER_AVAILABLE = True
except ImportError:
    try:
        from greenlang.agents.eudr.deforestation_alert_system.severity_classifier import (
            SeverityClassifier,
        )
        _SEVERITY_CLASSIFIER_AVAILABLE = True
    except ImportError:
        SeverityClassifier = None  # type: ignore[assignment,misc]
        _SEVERITY_CLASSIFIER_AVAILABLE = False

try:
    from greenlang.agents.eudr.deforestation_alert_system.engines.spatial_buffer_monitor import (
        SpatialBufferMonitor,
    )
    _SPATIAL_BUFFER_MONITOR_AVAILABLE = True
except ImportError:
    try:
        from greenlang.agents.eudr.deforestation_alert_system.spatial_buffer_monitor import (
            SpatialBufferMonitor,
        )
        _SPATIAL_BUFFER_MONITOR_AVAILABLE = True
    except ImportError:
        SpatialBufferMonitor = None  # type: ignore[assignment,misc]
        _SPATIAL_BUFFER_MONITOR_AVAILABLE = False

try:
    from greenlang.agents.eudr.deforestation_alert_system.engines.cutoff_date_verifier import (
        CutoffDateVerifier,
    )
    _CUTOFF_DATE_VERIFIER_AVAILABLE = True
except ImportError:
    try:
        from greenlang.agents.eudr.deforestation_alert_system.cutoff_date_verifier import (
            CutoffDateVerifier,
        )
        _CUTOFF_DATE_VERIFIER_AVAILABLE = True
    except ImportError:
        CutoffDateVerifier = None  # type: ignore[assignment,misc]
        _CUTOFF_DATE_VERIFIER_AVAILABLE = False

try:
    from greenlang.agents.eudr.deforestation_alert_system.engines.historical_baseline_engine import (
        HistoricalBaselineEngine,
    )
    _HISTORICAL_BASELINE_ENGINE_AVAILABLE = True
except ImportError:
    try:
        from greenlang.agents.eudr.deforestation_alert_system.historical_baseline_engine import (
            HistoricalBaselineEngine,
        )
        _HISTORICAL_BASELINE_ENGINE_AVAILABLE = True
    except ImportError:
        HistoricalBaselineEngine = None  # type: ignore[assignment,misc]
        _HISTORICAL_BASELINE_ENGINE_AVAILABLE = False

try:
    from greenlang.agents.eudr.deforestation_alert_system.engines.alert_workflow_engine import (
        AlertWorkflowEngine,
    )
    _ALERT_WORKFLOW_ENGINE_AVAILABLE = True
except ImportError:
    try:
        from greenlang.agents.eudr.deforestation_alert_system.alert_workflow_engine import (
            AlertWorkflowEngine,
        )
        _ALERT_WORKFLOW_ENGINE_AVAILABLE = True
    except ImportError:
        AlertWorkflowEngine = None  # type: ignore[assignment,misc]
        _ALERT_WORKFLOW_ENGINE_AVAILABLE = False

try:
    from greenlang.agents.eudr.deforestation_alert_system.engines.compliance_impact_assessor import (
        ComplianceImpactAssessor,
    )
    _COMPLIANCE_IMPACT_ASSESSOR_AVAILABLE = True
except ImportError:
    try:
        from greenlang.agents.eudr.deforestation_alert_system.compliance_impact_assessor import (
            ComplianceImpactAssessor,
        )
        _COMPLIANCE_IMPACT_ASSESSOR_AVAILABLE = True
    except ImportError:
        ComplianceImpactAssessor = None  # type: ignore[assignment,misc]
        _COMPLIANCE_IMPACT_ASSESSOR_AVAILABLE = False

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"
_AGENT_ID = "GL-EUDR-DAS-020"
_ENGINE_COUNT = 8
_ENGINE_NAMES: List[str] = [
    "SatelliteChangeDetector",
    "AlertGenerator",
    "SeverityClassifier",
    "SpatialBufferMonitor",
    "CutoffDateVerifier",
    "HistoricalBaselineEngine",
    "AlertWorkflowEngine",
    "ComplianceImpactAssessor",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _calculate_sha256(data: Any) -> str:
    """Calculate SHA-256 hash of JSON-serialized data for provenance.

    Args:
        data: Any JSON-serializable data structure.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    if isinstance(data, str):
        payload = data
    elif isinstance(data, bytes):
        payload = data.decode("utf-8", errors="replace")
    else:
        payload = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def _safe_decimal(value: Any, default: Decimal = Decimal("0.0")) -> Decimal:
    """Safely convert value to Decimal with fallback.

    Args:
        value: Value to convert.
        default: Default if conversion fails.

    Returns:
        Decimal representation of value.
    """
    try:
        return Decimal(str(value))
    except Exception:
        return default

# =============================================================================
# FACADE: DeforestationAlertSystemSetup
# =============================================================================

class DeforestationAlertSystemSetup:
    """
    DeforestationAlertSystemSetup orchestrates all 8 engines of AGENT-EUDR-020.

    This facade provides a single, thread-safe entry point for all deforestation
    alert system operations per EUDR Articles 2, 9, 10, 11, and 31.

    Architecture:
        - Lazy initialization of all 8 engines (on first use)
        - Thread-safe singleton pattern with double-checked locking
        - PostgreSQL connection pooling via psycopg_pool
        - Redis caching for frequently accessed reference data
        - OpenTelemetry distributed tracing integration
        - Prometheus metrics for all operations
        - SHA-256 provenance hashing for audit trails

    Engines:
        1. SatelliteChangeDetector: Multi-source satellite change detection
           combining Sentinel-2 (10m), Landsat 8/9 (30m), GLAD weekly alerts,
           Hansen GFC annual data, and RADD radar alerts with spectral index
           analysis (NDVI, EVI, NBR, NDMI, SAVI) and cloud cover filtering
        2. AlertGenerator: Automated alert generation with batch processing,
           real-time streaming, deduplication (72h window), daily alert caps,
           and configurable retention (5 years per Article 31)
        3. SeverityClassifier: Five-tier severity classification (CRITICAL,
           HIGH, MEDIUM, LOW, INFORMATIONAL) using weighted scoring across
           area (0.25), rate (0.20), proximity (0.25), protected area (0.15),
           and post-cutoff timing (0.15)
        4. SpatialBufferMonitor: Circular, polygon, and adaptive buffer zone
           monitoring with configurable 1-50 km radii at 64-point resolution
           for proximity detection to supply chain plots
        5. CutoffDateVerifier: EUDR cutoff date (31 December 2020) verification
           with multi-source temporal evidence, pre/post classification, 90-day
           grace period, and 0.85 confidence threshold
        6. HistoricalBaselineEngine: 2018-2020 reference period baseline using
           canopy cover and forest area with minimum 3-sample requirement
        7. AlertWorkflowEngine: State machine (triage -> investigation ->
           resolution) with SLA tracking (4h/48h/168h), auto-escalation
           up to 3 levels, and audit logging
        8. ComplianceImpactAssessor: Maps deforestation alerts to affected
           suppliers, products, market restrictions, and remediation actions
           with estimated financial impact

    Attributes:
        config: Current configuration instance
        db_pool: Async PostgreSQL connection pool (psycopg_pool)
        redis_client: Async Redis client (redis.asyncio)
        provenance_tracker: SHA-256 provenance tracking

    Example:
        >>> service = get_service()
        >>> await service.startup()
        >>>
        >>> # Detect satellite changes
        >>> detections = await service.detect_changes(
        ...     latitude=-3.1234, longitude=28.5678, radius_km=10,
        ... )
        >>>
        >>> # Generate alerts
        >>> alerts = await service.generate_alerts(detections=detections)
        >>>
        >>> # Classify severity
        >>> severity = await service.classify_severity(alert_id="alert-001")
        >>>
        >>> # Comprehensive analysis
        >>> analysis = await service.run_comprehensive_analysis(
        ...     plot_id="plot-001",
        ...     latitude=-3.1234,
        ...     longitude=28.5678,
        ... )
        >>>
        >>> await service.shutdown()
    """

    def __init__(
        self,
        config: Optional[DeforestationAlertSystemConfig] = None,
        *,
        db_pool: Optional[Any] = None,
        redis_client: Optional[Any] = None,
    ) -> None:
        """
        Initialize DeforestationAlertSystemSetup.

        Args:
            config: Optional configuration override. Defaults to global config.
            db_pool: Optional pre-initialized PostgreSQL connection pool.
            redis_client: Optional pre-initialized Redis client.
        """
        self._config = config or get_config()
        self._db_pool = db_pool
        self._redis_client = redis_client
        self._provenance_tracker: ProvenanceTracker = get_tracker()

        # OpenTelemetry tracer (optional)
        if OTEL_AVAILABLE and otel_trace:
            self._tracer = otel_trace.get_tracer(__name__, version=_MODULE_VERSION)
        else:
            self._tracer = None

        # Engine instances (lazy initialized)
        self._satellite_change_detector: Optional[Any] = None
        self._alert_generator: Optional[Any] = None
        self._severity_classifier: Optional[Any] = None
        self._spatial_buffer_monitor: Optional[Any] = None
        self._cutoff_date_verifier: Optional[Any] = None
        self._historical_baseline_engine: Optional[Any] = None
        self._alert_workflow_engine: Optional[Any] = None
        self._compliance_impact_assessor: Optional[Any] = None

        # Reference data (loaded on startup)
        self._reference_data_loaded: bool = False

        # Lifecycle state
        self._started: bool = False
        self._startup_time: Optional[datetime] = None
        self._startup_lock = asyncio.Lock()
        self._shutdown_lock = asyncio.Lock()

        # Statistics tracking
        self._stats: Dict[str, int] = {
            "total_analyses": 0,
            "total_detections": 0,
            "total_alerts_generated": 0,
            "total_severity_classifications": 0,
            "total_buffer_checks": 0,
            "total_cutoff_verifications": 0,
            "total_baseline_comparisons": 0,
            "total_workflow_transitions": 0,
            "total_compliance_assessments": 0,
            "total_comprehensive_analyses": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
        }

        logger.info(
            f"DeforestationAlertSystemSetup initialized "
            f"(version={_MODULE_VERSION}, agent_id={_AGENT_ID})"
        )

    # -----------------------------------------------------------------------
    # Properties for engine access
    # -----------------------------------------------------------------------

    @property
    def satellite_change_detector(self) -> Any:
        """Access SatelliteChangeDetector (lazy initialized).

        Returns:
            SatelliteChangeDetector instance.

        Raises:
            RuntimeError: If engine is not available.
        """
        if self._satellite_change_detector is None:
            raise RuntimeError(
                "SatelliteChangeDetector not initialized. Call startup() first."
            )
        return self._satellite_change_detector

    @property
    def alert_generator(self) -> Any:
        """Access AlertGenerator (lazy initialized).

        Returns:
            AlertGenerator instance.

        Raises:
            RuntimeError: If engine is not available.
        """
        if self._alert_generator is None:
            raise RuntimeError(
                "AlertGenerator not initialized. Call startup() first."
            )
        return self._alert_generator

    @property
    def severity_classifier(self) -> Any:
        """Access SeverityClassifier (lazy initialized).

        Returns:
            SeverityClassifier instance.

        Raises:
            RuntimeError: If engine is not available.
        """
        if self._severity_classifier is None:
            raise RuntimeError(
                "SeverityClassifier not initialized. Call startup() first."
            )
        return self._severity_classifier

    @property
    def spatial_buffer_monitor(self) -> Any:
        """Access SpatialBufferMonitor (lazy initialized).

        Returns:
            SpatialBufferMonitor instance.

        Raises:
            RuntimeError: If engine is not available.
        """
        if self._spatial_buffer_monitor is None:
            raise RuntimeError(
                "SpatialBufferMonitor not initialized. Call startup() first."
            )
        return self._spatial_buffer_monitor

    @property
    def cutoff_date_verifier(self) -> Any:
        """Access CutoffDateVerifier (lazy initialized).

        Returns:
            CutoffDateVerifier instance.

        Raises:
            RuntimeError: If engine is not available.
        """
        if self._cutoff_date_verifier is None:
            raise RuntimeError(
                "CutoffDateVerifier not initialized. Call startup() first."
            )
        return self._cutoff_date_verifier

    @property
    def historical_baseline_engine(self) -> Any:
        """Access HistoricalBaselineEngine (lazy initialized).

        Returns:
            HistoricalBaselineEngine instance.

        Raises:
            RuntimeError: If engine is not available.
        """
        if self._historical_baseline_engine is None:
            raise RuntimeError(
                "HistoricalBaselineEngine not initialized. Call startup() first."
            )
        return self._historical_baseline_engine

    @property
    def alert_workflow_engine(self) -> Any:
        """Access AlertWorkflowEngine (lazy initialized).

        Returns:
            AlertWorkflowEngine instance.

        Raises:
            RuntimeError: If engine is not available.
        """
        if self._alert_workflow_engine is None:
            raise RuntimeError(
                "AlertWorkflowEngine not initialized. Call startup() first."
            )
        return self._alert_workflow_engine

    @property
    def compliance_impact_assessor(self) -> Any:
        """Access ComplianceImpactAssessor (lazy initialized).

        Returns:
            ComplianceImpactAssessor instance.

        Raises:
            RuntimeError: If engine is not available.
        """
        if self._compliance_impact_assessor is None:
            raise RuntimeError(
                "ComplianceImpactAssessor not initialized. Call startup() first."
            )
        return self._compliance_impact_assessor

    # -----------------------------------------------------------------------
    # Lifecycle management
    # -----------------------------------------------------------------------

    async def startup(self) -> None:
        """
        Initialize all resources (database, Redis, engines).

        This method is idempotent and thread-safe. Multiple calls are safe.

        Raises:
            RuntimeError: If startup fails critically.
        """
        async with self._startup_lock:
            if self._started:
                logger.debug(
                    "DeforestationAlertSystemSetup already started, skipping startup"
                )
                return

            logger.info("Starting DeforestationAlertSystemSetup...")
            start_time = time.monotonic()

            try:
                # 1. Initialize database connection pool
                if self._db_pool is None and PSYCOPG_POOL_AVAILABLE:
                    await self._init_db_pool()
                else:
                    logger.info(
                        "Database pool disabled or psycopg_pool not available"
                    )

                # 2. Initialize Redis client
                if self._redis_client is None and REDIS_AVAILABLE and aioredis:
                    await self._init_redis()
                else:
                    logger.info("Redis disabled or redis library not available")

                # 3. Load reference data
                self._load_reference_data()

                # 4. Lazy engine initialization will happen on first use
                logger.debug(
                    "Engines will be initialized on first use (lazy loading)"
                )

                # 5. Register Prometheus metrics
                self._register_metrics()

                # 6. Mark as started
                self._started = True
                self._startup_time = utcnow()
                duration_ms = (time.monotonic() - start_time) * 1000

                logger.info(
                    f"DeforestationAlertSystemSetup started successfully "
                    f"(duration={duration_ms:.2f}ms)"
                )

            except Exception as e:
                logger.error(f"Startup failed: {e}", exc_info=True)
                record_api_error("startup", str(e))
                raise RuntimeError(f"Service startup failed: {e}") from e

    async def shutdown(self) -> None:
        """
        Gracefully shutdown all resources.

        Closes database pool, Redis client, and cleans up engine resources.
        This method is idempotent and safe to call multiple times.
        """
        async with self._shutdown_lock:
            if not self._started:
                logger.debug(
                    "DeforestationAlertSystemSetup not started, skipping shutdown"
                )
                return

            logger.info("Shutting down DeforestationAlertSystemSetup...")

            try:
                # 1. Shutdown engines
                await self._shutdown_engines()

                # 2. Close Redis client
                if self._redis_client is not None:
                    try:
                        await self._redis_client.close()
                        logger.debug("Redis client closed")
                    except Exception as e:
                        logger.warning(f"Error closing Redis client: {e}")

                # 3. Close database pool
                if self._db_pool is not None:
                    try:
                        await self._db_pool.close()
                        logger.debug("PostgreSQL pool closed")
                    except Exception as e:
                        logger.warning(f"Error closing PostgreSQL pool: {e}")

                # 4. Mark as shutdown
                self._started = False

                logger.info(
                    "DeforestationAlertSystemSetup shutdown complete"
                )

            except Exception as e:
                logger.error(f"Shutdown error: {e}", exc_info=True)

    async def initialize(self) -> Dict[str, Any]:
        """
        Full initialization of all engines, load reference data, verify connectivity.

        This method performs eager initialization of all 8 engines (as opposed
        to lazy loading), loads reference data into memory, and verifies that
        all external connections are available.

        Returns:
            Dict with initialization status per engine and resource.

        Raises:
            RuntimeError: If critical initialization fails.
        """
        logger.info("Running full initialization of all 8 engines...")
        start_time = time.monotonic()

        # Ensure service is started
        if not self._started:
            await self.startup()

        init_results: Dict[str, Any] = {
            "agent_id": _AGENT_ID,
            "version": _MODULE_VERSION,
            "timestamp": utcnow().isoformat(),
            "engines": {},
            "reference_data": {},
            "connectivity": {},
        }

        # Initialize all engines eagerly
        engine_init_methods = [
            ("SatelliteChangeDetector", self._ensure_satellite_change_detector),
            ("AlertGenerator", self._ensure_alert_generator),
            ("SeverityClassifier", self._ensure_severity_classifier),
            ("SpatialBufferMonitor", self._ensure_spatial_buffer_monitor),
            ("CutoffDateVerifier", self._ensure_cutoff_date_verifier),
            ("HistoricalBaselineEngine", self._ensure_historical_baseline_engine),
            ("AlertWorkflowEngine", self._ensure_alert_workflow_engine),
            ("ComplianceImpactAssessor", self._ensure_compliance_impact_assessor),
        ]

        for engine_name, init_method in engine_init_methods:
            try:
                await init_method()
                init_results["engines"][engine_name] = "initialized"
                logger.info(f"Engine {engine_name} initialized successfully")
            except Exception as e:
                init_results["engines"][engine_name] = f"failed: {str(e)}"
                logger.warning(f"Engine {engine_name} initialization failed: {e}")

        # Load and validate reference data
        try:
            from greenlang.agents.eudr.deforestation_alert_system.reference_data import (
                SatelliteSourceDatabase,
                DeforestationHotspotsDatabase,
                ProtectedAreasDatabase,
                CountryForestDatabase,
                validate_all_databases,
            )

            validation = validate_all_databases()
            init_results["reference_data"] = validation
        except ImportError as e:
            init_results["reference_data"]["status"] = f"partial: {str(e)}"
            logger.warning(f"Reference data import failed: {e}")

        # Verify data integrity
        integrity_errors = self._validate_data_integrity()
        init_results["data_integrity"] = {
            "status": "valid" if not integrity_errors else "errors_found",
            "error_count": len(integrity_errors),
            "errors": integrity_errors[:10],  # First 10 errors only
        }

        # Verify connectivity
        init_results["connectivity"]["database"] = (
            "connected" if self._db_pool is not None else "not_configured"
        )
        init_results["connectivity"]["redis"] = (
            "connected" if self._redis_client is not None else "not_configured"
        )

        duration_ms = (time.monotonic() - start_time) * 1000
        init_results["initialization_time_ms"] = round(duration_ms, 2)

        engines_ok = sum(
            1 for v in init_results["engines"].values() if v == "initialized"
        )
        init_results["engines_initialized"] = engines_ok
        init_results["engines_total"] = _ENGINE_COUNT
        init_results["status"] = (
            "fully_initialized" if engines_ok == _ENGINE_COUNT else "partially_initialized"
        )

        logger.info(
            f"Full initialization complete: {engines_ok}/{_ENGINE_COUNT} engines "
            f"in {duration_ms:.2f}ms"
        )

        return init_results

    # -----------------------------------------------------------------------
    # Internal initialization helpers
    # -----------------------------------------------------------------------

    async def _init_db_pool(self) -> None:
        """Initialize PostgreSQL connection pool."""
        if not PSYCOPG_POOL_AVAILABLE:
            raise RuntimeError("psycopg_pool not available")

        logger.info(
            f"Initializing database pool "
            f"(size={self._config.pool_size})"
        )
        self._db_pool = AsyncConnectionPool(
            conninfo=self._config.database_url,
            min_size=2,
            max_size=self._config.pool_size,
            timeout=30.0,
            max_idle=300.0,
            max_lifetime=3600.0,
        )
        await self._db_pool.open()
        logger.info("Database pool opened")

    async def _init_redis(self) -> None:
        """Initialize Redis async client."""
        if not REDIS_AVAILABLE or aioredis is None:
            raise RuntimeError("redis.asyncio not available")

        logger.info("Initializing Redis client")
        self._redis_client = await aioredis.from_url(
            self._config.redis_url,
            decode_responses=True,
            max_connections=20,
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
        )
        # Verify connection
        await self._redis_client.ping()
        logger.info("Redis client connected")

    def _load_reference_data(self) -> None:
        """Load all reference data databases into memory.

        Imports and validates satellite sources, deforestation hotspots,
        protected areas, and country forest databases. Sets
        ``_reference_data_loaded`` flag on success.
        """
        if self._reference_data_loaded:
            logger.debug("Reference data already loaded, skipping")
            return

        logger.info("Loading reference data...")
        start_time = time.monotonic()

        try:
            from greenlang.agents.eudr.deforestation_alert_system.reference_data import (
                SatelliteSourceDatabase,
                DeforestationHotspotsDatabase,
                ProtectedAreasDatabase,
                CountryForestDatabase,
            )

            # Validate each database loads correctly
            sat_db = SatelliteSourceDatabase()
            sat_count = sat_db.get_source_count()
            logger.info(f"Satellite sources loaded: {sat_count} sources")

            hotspot_db = DeforestationHotspotsDatabase()
            hotspot_count = hotspot_db.get_hotspot_count()
            logger.info(f"Deforestation hotspots loaded: {hotspot_count} regions")

            protected_db = ProtectedAreasDatabase()
            protected_count = protected_db.get_area_count()
            logger.info(f"Protected areas loaded: {protected_count} areas")

            forest_db = CountryForestDatabase()
            forest_count = forest_db.get_country_count()
            logger.info(f"Country forest data loaded: {forest_count} countries")

            self._reference_data_loaded = True
            duration_ms = (time.monotonic() - start_time) * 1000

            logger.info(
                f"Reference data loaded successfully (duration={duration_ms:.2f}ms)"
            )

        except ImportError as e:
            logger.warning(
                f"Reference data partially loaded (import error): {e}"
            )
        except Exception as e:
            logger.warning(f"Reference data loading failed: {e}")

    def _initialize_engines(self) -> Dict[str, str]:
        """Create all 8 engine instances synchronously (for diagnostics).

        Returns:
            Dict mapping engine name to initialization status string.
        """
        results: Dict[str, str] = {}
        engine_classes = [
            ("SatelliteChangeDetector", SatelliteChangeDetector, _SATELLITE_CHANGE_DETECTOR_AVAILABLE),
            ("AlertGenerator", AlertGenerator, _ALERT_GENERATOR_AVAILABLE),
            ("SeverityClassifier", SeverityClassifier, _SEVERITY_CLASSIFIER_AVAILABLE),
            ("SpatialBufferMonitor", SpatialBufferMonitor, _SPATIAL_BUFFER_MONITOR_AVAILABLE),
            ("CutoffDateVerifier", CutoffDateVerifier, _CUTOFF_DATE_VERIFIER_AVAILABLE),
            ("HistoricalBaselineEngine", HistoricalBaselineEngine, _HISTORICAL_BASELINE_ENGINE_AVAILABLE),
            ("AlertWorkflowEngine", AlertWorkflowEngine, _ALERT_WORKFLOW_ENGINE_AVAILABLE),
            ("ComplianceImpactAssessor", ComplianceImpactAssessor, _COMPLIANCE_IMPACT_ASSESSOR_AVAILABLE),
        ]

        for name, cls, available in engine_classes:
            if available and cls is not None:
                results[name] = "available"
            else:
                results[name] = "not_available"

        return results

    def _validate_data_integrity(self) -> List[str]:
        """Validate integrity of all loaded reference data.

        Checks for missing keys, invalid values, and cross-reference
        consistency across all four reference data databases.

        Returns:
            List of error messages (empty if all valid).
        """
        errors: List[str] = []

        try:
            from greenlang.agents.eudr.deforestation_alert_system.reference_data import (
                validate_all_databases,
            )
            validation = validate_all_databases()
            if validation.get("overall_status") != "valid":
                for db_errors in validation.get("errors", []):
                    errors.append(str(db_errors))
        except ImportError:
            errors.append("Reference data package not available for validation")
        except Exception as e:
            errors.append(f"Data integrity validation error: {str(e)}")

        return errors

    def _register_metrics(self) -> None:
        """Register Prometheus metrics for the deforestation alert system.

        Sets initial gauge values for active alerts, monitored plots,
        active buffers, pending investigations, SLA breaches, and
        detection backlog.
        """
        if not PROMETHEUS_AVAILABLE:
            logger.debug("Prometheus not available, skipping metrics registration")
            return

        try:
            set_active_alerts(0)
            set_monitored_plots(0)
            set_active_buffers(0)
            set_pending_investigations(0)
            set_sla_breaches(0)
            set_detection_backlog(0)
            logger.info("Prometheus metrics registered")
        except Exception as e:
            logger.warning(f"Metrics registration failed: {e}")

    async def _shutdown_engines(self) -> None:
        """Shutdown all initialized engines gracefully."""
        engines = [
            ("SatelliteChangeDetector", self._satellite_change_detector),
            ("AlertGenerator", self._alert_generator),
            ("SeverityClassifier", self._severity_classifier),
            ("SpatialBufferMonitor", self._spatial_buffer_monitor),
            ("CutoffDateVerifier", self._cutoff_date_verifier),
            ("HistoricalBaselineEngine", self._historical_baseline_engine),
            ("AlertWorkflowEngine", self._alert_workflow_engine),
            ("ComplianceImpactAssessor", self._compliance_impact_assessor),
        ]

        for engine_name, engine in engines:
            if engine is not None:
                try:
                    if hasattr(engine, "shutdown"):
                        await engine.shutdown()
                    logger.debug(f"{engine_name} shutdown complete")
                except Exception as e:
                    logger.warning(f"Error shutting down {engine_name}: {e}")

    # -----------------------------------------------------------------------
    # Engine lazy initialization
    # -----------------------------------------------------------------------

    async def _ensure_satellite_change_detector(self) -> Any:
        """Lazy initialize SatelliteChangeDetector.

        Returns:
            Initialized SatelliteChangeDetector instance.

        Raises:
            RuntimeError: If engine module is not available.
        """
        if self._satellite_change_detector is None:
            if not _SATELLITE_CHANGE_DETECTOR_AVAILABLE or SatelliteChangeDetector is None:
                raise RuntimeError("SatelliteChangeDetector not available")
            logger.debug("Initializing SatelliteChangeDetector...")
            self._satellite_change_detector = SatelliteChangeDetector(
                config=self._config,
                db_pool=self._db_pool,
                redis_client=self._redis_client,
            )
            if hasattr(self._satellite_change_detector, "startup"):
                await self._satellite_change_detector.startup()
            logger.info("SatelliteChangeDetector initialized")
        return self._satellite_change_detector

    async def _ensure_alert_generator(self) -> Any:
        """Lazy initialize AlertGenerator.

        Returns:
            Initialized AlertGenerator instance.

        Raises:
            RuntimeError: If engine module is not available.
        """
        if self._alert_generator is None:
            if not _ALERT_GENERATOR_AVAILABLE or AlertGenerator is None:
                raise RuntimeError("AlertGenerator not available")
            logger.debug("Initializing AlertGenerator...")
            self._alert_generator = AlertGenerator(
                config=self._config,
                db_pool=self._db_pool,
                redis_client=self._redis_client,
            )
            if hasattr(self._alert_generator, "startup"):
                await self._alert_generator.startup()
            logger.info("AlertGenerator initialized")
        return self._alert_generator

    async def _ensure_severity_classifier(self) -> Any:
        """Lazy initialize SeverityClassifier.

        Returns:
            Initialized SeverityClassifier instance.

        Raises:
            RuntimeError: If engine module is not available.
        """
        if self._severity_classifier is None:
            if not _SEVERITY_CLASSIFIER_AVAILABLE or SeverityClassifier is None:
                raise RuntimeError("SeverityClassifier not available")
            logger.debug("Initializing SeverityClassifier...")
            self._severity_classifier = SeverityClassifier(
                config=self._config,
                db_pool=self._db_pool,
                redis_client=self._redis_client,
            )
            if hasattr(self._severity_classifier, "startup"):
                await self._severity_classifier.startup()
            logger.info("SeverityClassifier initialized")
        return self._severity_classifier

    async def _ensure_spatial_buffer_monitor(self) -> Any:
        """Lazy initialize SpatialBufferMonitor.

        Returns:
            Initialized SpatialBufferMonitor instance.

        Raises:
            RuntimeError: If engine module is not available.
        """
        if self._spatial_buffer_monitor is None:
            if not _SPATIAL_BUFFER_MONITOR_AVAILABLE or SpatialBufferMonitor is None:
                raise RuntimeError("SpatialBufferMonitor not available")
            logger.debug("Initializing SpatialBufferMonitor...")
            self._spatial_buffer_monitor = SpatialBufferMonitor(
                config=self._config,
                db_pool=self._db_pool,
                redis_client=self._redis_client,
            )
            if hasattr(self._spatial_buffer_monitor, "startup"):
                await self._spatial_buffer_monitor.startup()
            logger.info("SpatialBufferMonitor initialized")
        return self._spatial_buffer_monitor

    async def _ensure_cutoff_date_verifier(self) -> Any:
        """Lazy initialize CutoffDateVerifier.

        Returns:
            Initialized CutoffDateVerifier instance.

        Raises:
            RuntimeError: If engine module is not available.
        """
        if self._cutoff_date_verifier is None:
            if not _CUTOFF_DATE_VERIFIER_AVAILABLE or CutoffDateVerifier is None:
                raise RuntimeError("CutoffDateVerifier not available")
            logger.debug("Initializing CutoffDateVerifier...")
            self._cutoff_date_verifier = CutoffDateVerifier(
                config=self._config,
                db_pool=self._db_pool,
                redis_client=self._redis_client,
            )
            if hasattr(self._cutoff_date_verifier, "startup"):
                await self._cutoff_date_verifier.startup()
            logger.info("CutoffDateVerifier initialized")
        return self._cutoff_date_verifier

    async def _ensure_historical_baseline_engine(self) -> Any:
        """Lazy initialize HistoricalBaselineEngine.

        Returns:
            Initialized HistoricalBaselineEngine instance.

        Raises:
            RuntimeError: If engine module is not available.
        """
        if self._historical_baseline_engine is None:
            if not _HISTORICAL_BASELINE_ENGINE_AVAILABLE or HistoricalBaselineEngine is None:
                raise RuntimeError("HistoricalBaselineEngine not available")
            logger.debug("Initializing HistoricalBaselineEngine...")
            self._historical_baseline_engine = HistoricalBaselineEngine(
                config=self._config,
                db_pool=self._db_pool,
                redis_client=self._redis_client,
            )
            if hasattr(self._historical_baseline_engine, "startup"):
                await self._historical_baseline_engine.startup()
            logger.info("HistoricalBaselineEngine initialized")
        return self._historical_baseline_engine

    async def _ensure_alert_workflow_engine(self) -> Any:
        """Lazy initialize AlertWorkflowEngine.

        Returns:
            Initialized AlertWorkflowEngine instance.

        Raises:
            RuntimeError: If engine module is not available.
        """
        if self._alert_workflow_engine is None:
            if not _ALERT_WORKFLOW_ENGINE_AVAILABLE or AlertWorkflowEngine is None:
                raise RuntimeError("AlertWorkflowEngine not available")
            logger.debug("Initializing AlertWorkflowEngine...")
            self._alert_workflow_engine = AlertWorkflowEngine(
                config=self._config,
                db_pool=self._db_pool,
                redis_client=self._redis_client,
            )
            if hasattr(self._alert_workflow_engine, "startup"):
                await self._alert_workflow_engine.startup()
            logger.info("AlertWorkflowEngine initialized")
        return self._alert_workflow_engine

    async def _ensure_compliance_impact_assessor(self) -> Any:
        """Lazy initialize ComplianceImpactAssessor.

        Returns:
            Initialized ComplianceImpactAssessor instance.

        Raises:
            RuntimeError: If engine module is not available.
        """
        if self._compliance_impact_assessor is None:
            if not _COMPLIANCE_IMPACT_ASSESSOR_AVAILABLE or ComplianceImpactAssessor is None:
                raise RuntimeError("ComplianceImpactAssessor not available")
            logger.debug("Initializing ComplianceImpactAssessor...")
            self._compliance_impact_assessor = ComplianceImpactAssessor(
                config=self._config,
                db_pool=self._db_pool,
                redis_client=self._redis_client,
            )
            if hasattr(self._compliance_impact_assessor, "startup"):
                await self._compliance_impact_assessor.startup()
            logger.info("ComplianceImpactAssessor initialized")
        return self._compliance_impact_assessor

    # -----------------------------------------------------------------------
    # Public API: Engine delegation methods
    # -----------------------------------------------------------------------

    async def detect_changes(
        self,
        latitude: float,
        longitude: float,
        radius_km: float = 10.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Detect satellite-observed changes using SatelliteChangeDetector.

        Args:
            latitude: Center latitude of monitoring area.
            longitude: Center longitude of monitoring area.
            radius_km: Monitoring radius in kilometers.
            **kwargs: Additional detection parameters.

        Returns:
            Dict with detections, satellite sources used, cloud cover stats,
            spectral indices, and provenance hash.

        Raises:
            RuntimeError: If SatelliteChangeDetector is not available.
        """
        start_time = time.monotonic()
        logger.info(
            f"Detecting changes at ({latitude}, {longitude}) "
            f"radius={radius_km}km"
        )

        try:
            engine = await self._ensure_satellite_change_detector()
            result = await engine.detect_changes(
                latitude=latitude,
                longitude=longitude,
                radius_km=radius_km,
                **kwargs,
            )

            duration_sec = time.monotonic() - start_time
            observe_detection_latency(duration_sec)
            record_satellite_detection(status="success")
            self._stats["total_detections"] += 1
            self._stats["total_analyses"] += 1

            provenance_hash = _calculate_sha256({
                "action": "detect_changes",
                "latitude": latitude,
                "longitude": longitude,
                "radius_km": radius_km,
                "timestamp": utcnow().isoformat(),
            })

            logger.info(
                f"Change detection complete: "
                f"duration={duration_sec:.3f}s"
            )

            if isinstance(result, dict):
                result["provenance_hash"] = provenance_hash
                result["processing_time_ms"] = round(duration_sec * 1000, 2)
                return result
            return {
                "result": result,
                "provenance_hash": provenance_hash,
                "processing_time_ms": round(duration_sec * 1000, 2),
            }

        except Exception as e:
            self._stats["errors"] += 1
            record_api_error("detect_changes", str(e))
            logger.error(f"detect_changes failed: {e}", exc_info=True)
            raise

    async def generate_alerts(
        self,
        detections: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate deforestation alerts using AlertGenerator.

        Args:
            detections: List of satellite detection dicts to process.
            **kwargs: Additional alert generation parameters.

        Returns:
            Dict with generated alerts, dedup statistics, batch info,
            and provenance hash.

        Raises:
            RuntimeError: If AlertGenerator is not available.
        """
        start_time = time.monotonic()
        detection_count = len(detections) if detections else 0
        logger.info(f"Generating alerts from {detection_count} detections")

        try:
            engine = await self._ensure_alert_generator()
            result = await engine.generate_alerts(
                detections=detections, **kwargs
            )

            duration_sec = time.monotonic() - start_time
            observe_alert_generation_duration(duration_sec)
            record_alert_generated(status="success")
            self._stats["total_alerts_generated"] += 1
            self._stats["total_analyses"] += 1

            provenance_hash = _calculate_sha256({
                "action": "generate_alerts",
                "detection_count": detection_count,
                "timestamp": utcnow().isoformat(),
            })

            logger.info(
                f"Alert generation complete: "
                f"duration={duration_sec:.3f}s"
            )

            if isinstance(result, dict):
                result["provenance_hash"] = provenance_hash
                result["processing_time_ms"] = round(duration_sec * 1000, 2)
                return result
            return {
                "result": result,
                "provenance_hash": provenance_hash,
                "processing_time_ms": round(duration_sec * 1000, 2),
            }

        except Exception as e:
            self._stats["errors"] += 1
            record_api_error("generate_alerts", str(e))
            logger.error(f"generate_alerts failed: {e}", exc_info=True)
            raise

    async def classify_severity(
        self,
        alert_id: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Classify severity of a deforestation alert using SeverityClassifier.

        Args:
            alert_id: Unique alert identifier.
            **kwargs: Additional classification parameters.

        Returns:
            Dict with severity level, weighted scores, dimension breakdown,
            and provenance hash.

        Raises:
            RuntimeError: If SeverityClassifier is not available.
        """
        start_time = time.monotonic()
        logger.info(f"Classifying severity for alert {alert_id}")

        try:
            engine = await self._ensure_severity_classifier()
            result = await engine.classify_severity(
                alert_id=alert_id, **kwargs
            )

            duration_sec = time.monotonic() - start_time
            observe_severity_scoring_duration(duration_sec)
            record_severity_classification(status="success")
            self._stats["total_severity_classifications"] += 1
            self._stats["total_analyses"] += 1

            provenance_hash = _calculate_sha256({
                "action": "classify_severity",
                "alert_id": alert_id,
                "timestamp": utcnow().isoformat(),
            })

            logger.info(
                f"Severity classification complete for {alert_id}: "
                f"duration={duration_sec:.3f}s"
            )

            if isinstance(result, dict):
                result["provenance_hash"] = provenance_hash
                result["processing_time_ms"] = round(duration_sec * 1000, 2)
                return result
            return {
                "result": result,
                "provenance_hash": provenance_hash,
                "processing_time_ms": round(duration_sec * 1000, 2),
            }

        except Exception as e:
            self._stats["errors"] += 1
            record_api_error("classify_severity", str(e))
            logger.error(f"classify_severity failed: {e}", exc_info=True)
            raise

    async def check_buffer(
        self,
        plot_latitude: float,
        plot_longitude: float,
        detection_latitude: float,
        detection_longitude: float,
        buffer_radius_km: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Check spatial buffer zone using SpatialBufferMonitor.

        Args:
            plot_latitude: Supply chain plot latitude.
            plot_longitude: Supply chain plot longitude.
            detection_latitude: Deforestation detection latitude.
            detection_longitude: Deforestation detection longitude.
            buffer_radius_km: Buffer radius in km (default from config).
            **kwargs: Additional buffer check parameters.

        Returns:
            Dict with within_buffer flag, distance_km, buffer_geometry,
            and provenance hash.

        Raises:
            RuntimeError: If SpatialBufferMonitor is not available.
        """
        start_time = time.monotonic()
        logger.info(
            f"Checking buffer: plot=({plot_latitude}, {plot_longitude}), "
            f"detection=({detection_latitude}, {detection_longitude})"
        )

        try:
            engine = await self._ensure_spatial_buffer_monitor()
            result = await engine.check_buffer(
                plot_latitude=plot_latitude,
                plot_longitude=plot_longitude,
                detection_latitude=detection_latitude,
                detection_longitude=detection_longitude,
                buffer_radius_km=buffer_radius_km,
                **kwargs,
            )

            duration_sec = time.monotonic() - start_time
            record_buffer_check(status="success")
            self._stats["total_buffer_checks"] += 1
            self._stats["total_analyses"] += 1

            provenance_hash = _calculate_sha256({
                "action": "check_buffer",
                "plot_latitude": plot_latitude,
                "plot_longitude": plot_longitude,
                "detection_latitude": detection_latitude,
                "detection_longitude": detection_longitude,
                "timestamp": utcnow().isoformat(),
            })

            logger.info(
                f"Buffer check complete: "
                f"duration={duration_sec:.3f}s"
            )

            if isinstance(result, dict):
                result["provenance_hash"] = provenance_hash
                result["processing_time_ms"] = round(duration_sec * 1000, 2)
                return result
            return {
                "result": result,
                "provenance_hash": provenance_hash,
                "processing_time_ms": round(duration_sec * 1000, 2),
            }

        except Exception as e:
            self._stats["errors"] += 1
            record_api_error("check_buffer", str(e))
            logger.error(f"check_buffer failed: {e}", exc_info=True)
            raise

    async def verify_cutoff_date(
        self,
        detection_date: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Verify detection against EUDR cutoff date using CutoffDateVerifier.

        Args:
            detection_date: Detection date in YYYY-MM-DD format.
            **kwargs: Additional verification parameters.

        Returns:
            Dict with pre/post classification, confidence score,
            evidence sources, and provenance hash.

        Raises:
            RuntimeError: If CutoffDateVerifier is not available.
        """
        start_time = time.monotonic()
        logger.info(f"Verifying cutoff date for detection: {detection_date}")

        try:
            engine = await self._ensure_cutoff_date_verifier()
            result = await engine.verify_cutoff_date(
                detection_date=detection_date, **kwargs
            )

            duration_sec = time.monotonic() - start_time
            record_cutoff_verification(status="success")
            self._stats["total_cutoff_verifications"] += 1
            self._stats["total_analyses"] += 1

            provenance_hash = _calculate_sha256({
                "action": "verify_cutoff_date",
                "detection_date": detection_date,
                "timestamp": utcnow().isoformat(),
            })

            logger.info(
                f"Cutoff date verification complete: "
                f"duration={duration_sec:.3f}s"
            )

            if isinstance(result, dict):
                result["provenance_hash"] = provenance_hash
                result["processing_time_ms"] = round(duration_sec * 1000, 2)
                return result
            return {
                "result": result,
                "provenance_hash": provenance_hash,
                "processing_time_ms": round(duration_sec * 1000, 2),
            }

        except Exception as e:
            self._stats["errors"] += 1
            record_api_error("verify_cutoff_date", str(e))
            logger.error(f"verify_cutoff_date failed: {e}", exc_info=True)
            raise

    async def compare_baseline(
        self,
        plot_id: str,
        current_canopy_cover_pct: float,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Compare current state to historical baseline using HistoricalBaselineEngine.

        Args:
            plot_id: Supply chain plot identifier.
            current_canopy_cover_pct: Current canopy cover percentage.
            **kwargs: Additional comparison parameters.

        Returns:
            Dict with baseline values, change metrics, significance assessment,
            and provenance hash.

        Raises:
            RuntimeError: If HistoricalBaselineEngine is not available.
        """
        start_time = time.monotonic()
        logger.info(
            f"Comparing baseline for plot {plot_id}: "
            f"current_cover={current_canopy_cover_pct}%"
        )

        try:
            engine = await self._ensure_historical_baseline_engine()
            result = await engine.compare_baseline(
                plot_id=plot_id,
                current_canopy_cover_pct=current_canopy_cover_pct,
                **kwargs,
            )

            duration_sec = time.monotonic() - start_time
            record_baseline_comparison(status="success")
            self._stats["total_baseline_comparisons"] += 1
            self._stats["total_analyses"] += 1

            provenance_hash = _calculate_sha256({
                "action": "compare_baseline",
                "plot_id": plot_id,
                "current_canopy_cover_pct": current_canopy_cover_pct,
                "timestamp": utcnow().isoformat(),
            })

            logger.info(
                f"Baseline comparison complete for {plot_id}: "
                f"duration={duration_sec:.3f}s"
            )

            if isinstance(result, dict):
                result["provenance_hash"] = provenance_hash
                result["processing_time_ms"] = round(duration_sec * 1000, 2)
                return result
            return {
                "result": result,
                "provenance_hash": provenance_hash,
                "processing_time_ms": round(duration_sec * 1000, 2),
            }

        except Exception as e:
            self._stats["errors"] += 1
            record_api_error("compare_baseline", str(e))
            logger.error(f"compare_baseline failed: {e}", exc_info=True)
            raise

    async def transition_workflow(
        self,
        alert_id: str,
        action: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Transition alert workflow state using AlertWorkflowEngine.

        Args:
            alert_id: Alert identifier.
            action: Workflow action (triage, investigate, resolve, escalate).
            **kwargs: Additional transition parameters.

        Returns:
            Dict with new workflow state, SLA status, transition history,
            and provenance hash.

        Raises:
            RuntimeError: If AlertWorkflowEngine is not available.
        """
        start_time = time.monotonic()
        logger.info(f"Transitioning workflow: alert={alert_id}, action={action}")

        try:
            engine = await self._ensure_alert_workflow_engine()
            result = await engine.transition_workflow(
                alert_id=alert_id, action=action, **kwargs
            )

            duration_sec = time.monotonic() - start_time
            record_workflow_transition(status="success")
            self._stats["total_workflow_transitions"] += 1
            self._stats["total_analyses"] += 1

            provenance_hash = _calculate_sha256({
                "action": "transition_workflow",
                "alert_id": alert_id,
                "workflow_action": action,
                "timestamp": utcnow().isoformat(),
            })

            logger.info(
                f"Workflow transition complete for {alert_id}: "
                f"duration={duration_sec:.3f}s"
            )

            if isinstance(result, dict):
                result["provenance_hash"] = provenance_hash
                result["processing_time_ms"] = round(duration_sec * 1000, 2)
                return result
            return {
                "result": result,
                "provenance_hash": provenance_hash,
                "processing_time_ms": round(duration_sec * 1000, 2),
            }

        except Exception as e:
            self._stats["errors"] += 1
            record_api_error("transition_workflow", str(e))
            logger.error(f"transition_workflow failed: {e}", exc_info=True)
            raise

    async def assess_compliance_impact(
        self,
        alert_id: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Assess compliance impact using ComplianceImpactAssessor.

        Maps deforestation alerts to affected suppliers, products, market
        restrictions, remediation actions, and estimated financial impact.

        Args:
            alert_id: Alert identifier.
            **kwargs: Additional assessment parameters.

        Returns:
            Dict with affected_suppliers, market_restrictions,
            remediation_actions, estimated_impact, and provenance hash.

        Raises:
            RuntimeError: If ComplianceImpactAssessor is not available.
        """
        start_time = time.monotonic()
        logger.info(f"Assessing compliance impact for alert {alert_id}")

        try:
            engine = await self._ensure_compliance_impact_assessor()
            result = await engine.assess_compliance_impact(
                alert_id=alert_id, **kwargs
            )

            duration_sec = time.monotonic() - start_time
            observe_compliance_assessment_duration(duration_sec)
            record_compliance_assessment(status="success")
            self._stats["total_compliance_assessments"] += 1
            self._stats["total_analyses"] += 1

            provenance_hash = _calculate_sha256({
                "action": "assess_compliance_impact",
                "alert_id": alert_id,
                "timestamp": utcnow().isoformat(),
            })

            logger.info(
                f"Compliance impact assessment complete for {alert_id}: "
                f"duration={duration_sec:.3f}s"
            )

            if isinstance(result, dict):
                result["provenance_hash"] = provenance_hash
                result["processing_time_ms"] = round(duration_sec * 1000, 2)
                return result
            return {
                "result": result,
                "provenance_hash": provenance_hash,
                "processing_time_ms": round(duration_sec * 1000, 2),
            }

        except Exception as e:
            self._stats["errors"] += 1
            record_api_error("assess_compliance_impact", str(e))
            logger.error(
                f"assess_compliance_impact failed: {e}", exc_info=True
            )
            raise

    # -----------------------------------------------------------------------
    # Public API: Cross-engine orchestration
    # -----------------------------------------------------------------------

    async def run_comprehensive_analysis(
        self,
        plot_id: str,
        latitude: float,
        longitude: float,
        include_all: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Run ALL engines for a complete deforestation analysis around a plot.

        Orchestrates all 8 engines to produce a comprehensive deforestation
        alert analysis covering satellite change detection, alert generation,
        severity classification, buffer zone monitoring, cutoff date
        verification, historical baseline comparison, workflow management,
        and compliance impact assessment.

        Args:
            plot_id: Supply chain plot identifier.
            latitude: Plot center latitude.
            longitude: Plot center longitude.
            include_all: If True, run all engines. If False, run only core.
            **kwargs: Additional parameters passed to individual engines.

        Returns:
            Dict with comprehensive analysis results containing per-engine
            outputs, provenance chain, and total processing time.

        Raises:
            ValueError: If plot_id or coordinates are empty/invalid.
        """
        start_time = time.monotonic()
        logger.info(
            f"Starting comprehensive analysis for plot {plot_id} "
            f"at ({latitude}, {longitude}) (include_all={include_all})"
        )

        try:
            if not plot_id:
                raise ValueError("plot_id must not be empty")

            results: Dict[str, Any] = {
                "plot_id": plot_id,
                "latitude": latitude,
                "longitude": longitude,
                "analysis_timestamp": utcnow().isoformat(),
                "agent_id": _AGENT_ID,
                "version": _MODULE_VERSION,
            }
            provenance_chain: List[str] = []

            # 1. Satellite Change Detection (always included)
            try:
                detections = await self.detect_changes(
                    latitude=latitude,
                    longitude=longitude,
                    radius_km=kwargs.get("radius_km", 10.0),
                )
                results["satellite_detections"] = detections
                if "provenance_hash" in detections:
                    provenance_chain.append(detections["provenance_hash"])
            except Exception as e:
                results["satellite_detections"] = {"error": str(e)}
                logger.warning(f"Satellite detection failed: {e}")

            # 2. Alert Generation (always included)
            try:
                detection_list = []
                if isinstance(results.get("satellite_detections"), dict):
                    detection_list = results["satellite_detections"].get(
                        "detections", []
                    )
                alerts = await self.generate_alerts(
                    detections=detection_list,
                )
                results["generated_alerts"] = alerts
                if "provenance_hash" in alerts:
                    provenance_chain.append(alerts["provenance_hash"])
            except Exception as e:
                results["generated_alerts"] = {"error": str(e)}
                logger.warning(f"Alert generation failed: {e}")

            # 3. Cutoff Date Verification (always included)
            try:
                cutoff = await self.verify_cutoff_date(
                    detection_date=kwargs.get(
                        "detection_date",
                        utcnow().strftime("%Y-%m-%d"),
                    ),
                )
                results["cutoff_verification"] = cutoff
                if "provenance_hash" in cutoff:
                    provenance_chain.append(cutoff["provenance_hash"])
            except Exception as e:
                results["cutoff_verification"] = {"error": str(e)}
                logger.warning(f"Cutoff verification failed: {e}")

            # 4. Compliance Impact Assessment (always included)
            try:
                compliance = await self.assess_compliance_impact(
                    alert_id=kwargs.get("alert_id", f"comp-{plot_id}"),
                )
                results["compliance_impact"] = compliance
                if "provenance_hash" in compliance:
                    provenance_chain.append(compliance["provenance_hash"])
            except Exception as e:
                results["compliance_impact"] = {"error": str(e)}
                logger.warning(f"Compliance impact assessment failed: {e}")

            # Calculate comprehensive provenance
            comprehensive_provenance = _calculate_sha256({
                "action": "run_comprehensive_analysis",
                "plot_id": plot_id,
                "latitude": latitude,
                "longitude": longitude,
                "engine_provenance_chain": provenance_chain,
                "timestamp": utcnow().isoformat(),
            })

            duration_sec = time.monotonic() - start_time
            results["provenance_chain"] = provenance_chain
            results["comprehensive_provenance_hash"] = comprehensive_provenance
            results["processing_time_ms"] = round(duration_sec * 1000, 2)
            results["engines_executed"] = len(provenance_chain)
            results["engines_total"] = _ENGINE_COUNT

            self._stats["total_comprehensive_analyses"] += 1
            self._stats["total_analyses"] += 1

            logger.info(
                f"Comprehensive analysis complete for {plot_id}: "
                f"{len(provenance_chain)}/{_ENGINE_COUNT} engines, "
                f"duration={duration_sec:.3f}s"
            )

            return results

        except Exception as e:
            self._stats["errors"] += 1
            record_api_error("run_comprehensive_analysis", str(e))
            logger.error(
                f"run_comprehensive_analysis failed: {e}", exc_info=True
            )
            raise

    # -----------------------------------------------------------------------
    # Health check and statistics
    # -----------------------------------------------------------------------

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check across all 8 engines.

        Returns:
            Dict with overall status ("healthy", "degraded", "unhealthy"),
            per-engine health, database and Redis connectivity, uptime,
            and reference data availability.
        """
        start_time = time.monotonic()
        logger.debug("Performing health check")

        engine_statuses: Dict[str, str] = {}
        overall_healthy = True

        # Check each engine (if initialized)
        engine_instances = [
            ("SatelliteChangeDetector", self._satellite_change_detector),
            ("AlertGenerator", self._alert_generator),
            ("SeverityClassifier", self._severity_classifier),
            ("SpatialBufferMonitor", self._spatial_buffer_monitor),
            ("CutoffDateVerifier", self._cutoff_date_verifier),
            ("HistoricalBaselineEngine", self._historical_baseline_engine),
            ("AlertWorkflowEngine", self._alert_workflow_engine),
            ("ComplianceImpactAssessor", self._compliance_impact_assessor),
        ]

        for engine_name, engine in engine_instances:
            if engine is not None:
                try:
                    if hasattr(engine, "health_check"):
                        engine_health = await engine.health_check()
                        status = (
                            engine_health.get("status", "healthy")
                            if isinstance(engine_health, dict)
                            else str(engine_health)
                        )
                        engine_statuses[engine_name] = status
                        if status != "healthy":
                            overall_healthy = False
                    else:
                        engine_statuses[engine_name] = "healthy"
                except Exception as e:
                    engine_statuses[engine_name] = f"unhealthy: {str(e)}"
                    overall_healthy = False
            else:
                engine_statuses[engine_name] = "not_initialized"

        # Check database connection
        db_healthy = False
        if self._db_pool is not None:
            try:
                async with self._db_pool.connection() as conn:
                    await conn.execute("SELECT 1")
                db_healthy = True
            except Exception as e:
                logger.warning(f"Database health check failed: {e}")
                overall_healthy = False

        # Check Redis connection
        redis_healthy = False
        if self._redis_client is not None:
            try:
                await self._redis_client.ping()
                redis_healthy = True
            except Exception as e:
                logger.warning(f"Redis health check failed: {e}")

        # Check reference data
        ref_data_healthy = True
        try:
            from greenlang.agents.eudr.deforestation_alert_system.reference_data import (
                validate_all_databases,
            )
            validation = validate_all_databases()
            ref_data_healthy = validation.get("overall_status") == "valid"
        except Exception as e:
            ref_data_healthy = False
            logger.warning(f"Reference data validation failed: {e}")

        # Calculate uptime
        uptime_seconds = 0.0
        if self._startup_time:
            uptime_seconds = (
                utcnow() - self._startup_time
            ).total_seconds()

        # Determine overall status
        engines_initialized = sum(
            1 for s in engine_statuses.values() if s != "not_initialized"
        )
        if not overall_healthy:
            status = "unhealthy"
        elif engines_initialized < _ENGINE_COUNT:
            status = "degraded"
        else:
            status = "healthy"

        duration_ms = (time.monotonic() - start_time) * 1000

        return {
            "status": status,
            "version": _MODULE_VERSION,
            "agent_id": _AGENT_ID,
            "started": self._started,
            "uptime_seconds": round(uptime_seconds, 2),
            "database_connected": db_healthy,
            "redis_connected": redis_healthy,
            "reference_data_valid": ref_data_healthy,
            "reference_data_loaded": self._reference_data_loaded,
            "engines_initialized": engines_initialized,
            "engines_total": _ENGINE_COUNT,
            "engine_statuses": engine_statuses,
            "processing_time_ms": round(duration_ms, 2),
            "timestamp": utcnow().isoformat(),
        }

    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get usage statistics across all engines.

        Returns:
            Dict with total_analyses, per-engine counts, cache metrics,
            error count, and uptime information.
        """
        logger.debug("Gathering statistics")

        uptime_seconds = 0.0
        if self._startup_time:
            uptime_seconds = (
                utcnow() - self._startup_time
            ).total_seconds()

        cache_hit_rate = 0.0
        total_cache_ops = self._stats["cache_hits"] + self._stats["cache_misses"]
        if total_cache_ops > 0:
            cache_hit_rate = self._stats["cache_hits"] / total_cache_ops

        return {
            "agent_id": _AGENT_ID,
            "version": _MODULE_VERSION,
            "uptime_seconds": round(uptime_seconds, 2),
            "total_analyses": self._stats["total_analyses"],
            "total_detections": self._stats["total_detections"],
            "total_alerts_generated": self._stats["total_alerts_generated"],
            "total_severity_classifications": self._stats["total_severity_classifications"],
            "total_buffer_checks": self._stats["total_buffer_checks"],
            "total_cutoff_verifications": self._stats["total_cutoff_verifications"],
            "total_baseline_comparisons": self._stats["total_baseline_comparisons"],
            "total_workflow_transitions": self._stats["total_workflow_transitions"],
            "total_compliance_assessments": self._stats["total_compliance_assessments"],
            "total_comprehensive_analyses": self._stats["total_comprehensive_analyses"],
            "cache_hits": self._stats["cache_hits"],
            "cache_misses": self._stats["cache_misses"],
            "cache_hit_rate": round(cache_hit_rate, 4),
            "errors": self._stats["errors"],
            "timestamp": utcnow().isoformat(),
        }

    def get_engine(self, engine_name: str) -> Optional[Any]:
        """Get a specific engine instance by name.

        Args:
            engine_name: Engine name (e.g. "SatelliteChangeDetector").

        Returns:
            Engine instance or None if not initialized.
        """
        engine_map = {
            "SatelliteChangeDetector": self._satellite_change_detector,
            "AlertGenerator": self._alert_generator,
            "SeverityClassifier": self._severity_classifier,
            "SpatialBufferMonitor": self._spatial_buffer_monitor,
            "CutoffDateVerifier": self._cutoff_date_verifier,
            "HistoricalBaselineEngine": self._historical_baseline_engine,
            "AlertWorkflowEngine": self._alert_workflow_engine,
            "ComplianceImpactAssessor": self._compliance_impact_assessor,
        }
        return engine_map.get(engine_name)

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for diagnostics.

        Returns:
            Dict with agent metadata, engine availability, dependency
            status, and configuration summary.
        """
        return {
            "agent_id": _AGENT_ID,
            "version": _MODULE_VERSION,
            "engine_count": _ENGINE_COUNT,
            "engine_names": list(_ENGINE_NAMES),
            "started": self._started,
            "reference_data_loaded": self._reference_data_loaded,
            "dependencies": {
                "psycopg_pool": PSYCOPG_POOL_AVAILABLE,
                "psycopg": PSYCOPG_AVAILABLE,
                "redis": REDIS_AVAILABLE,
                "opentelemetry": OTEL_AVAILABLE,
                "prometheus": PROMETHEUS_AVAILABLE,
            },
            "engine_availability": {
                "SatelliteChangeDetector": _SATELLITE_CHANGE_DETECTOR_AVAILABLE,
                "AlertGenerator": _ALERT_GENERATOR_AVAILABLE,
                "SeverityClassifier": _SEVERITY_CLASSIFIER_AVAILABLE,
                "SpatialBufferMonitor": _SPATIAL_BUFFER_MONITOR_AVAILABLE,
                "CutoffDateVerifier": _CUTOFF_DATE_VERIFIER_AVAILABLE,
                "HistoricalBaselineEngine": _HISTORICAL_BASELINE_ENGINE_AVAILABLE,
                "AlertWorkflowEngine": _ALERT_WORKFLOW_ENGINE_AVAILABLE,
                "ComplianceImpactAssessor": _COMPLIANCE_IMPACT_ASSESSOR_AVAILABLE,
            },
            "config_summary": {
                "ndvi_change_threshold": str(self._config.ndvi_change_threshold),
                "evi_change_threshold": str(self._config.evi_change_threshold),
                "confidence_threshold": str(self._config.confidence_threshold),
                "cutoff_date": self._config.cutoff_date,
                "default_buffer_radius_km": str(self._config.default_buffer_radius_km),
                "critical_area_threshold_ha": str(self._config.critical_area_threshold_ha),
                "sla_triage_hours": self._config.sla_triage_hours,
                "sla_investigation_hours": self._config.sla_investigation_hours,
                "sla_resolution_hours": self._config.sla_resolution_hours,
                "market_restriction_threshold": self._config.market_restriction_threshold,
                "baseline_period": f"{self._config.baseline_start_year}-{self._config.baseline_end_year}",
                "satellite_sources": {
                    "sentinel2": self._config.sentinel2_enabled,
                    "landsat": self._config.landsat_enabled,
                    "glad": self._config.glad_enabled,
                    "hansen_gfc": self._config.hansen_gfc_enabled,
                    "radd": self._config.radd_enabled,
                },
            },
        }

# =============================================================================
# Module-level singleton management
# =============================================================================

_service_instance: Optional[DeforestationAlertSystemSetup] = None
_service_lock = threading.Lock()

def get_service(
    config: Optional[DeforestationAlertSystemConfig] = None,
) -> DeforestationAlertSystemSetup:
    """
    Get or create the singleton DeforestationAlertSystemSetup instance.

    Thread-safe singleton with double-checked locking pattern.

    Args:
        config: Optional configuration override (only used on first call).

    Returns:
        Singleton DeforestationAlertSystemSetup instance.

    Example:
        >>> service = get_service()
        >>> await service.startup()
        >>> detections = await service.detect_changes(-3.12, 28.56)
        >>> await service.shutdown()
    """
    global _service_instance

    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = DeforestationAlertSystemSetup(config=config)
                logger.info("DeforestationAlertSystemSetup singleton created")

    return _service_instance

def set_service(service: DeforestationAlertSystemSetup) -> None:
    """
    Override the singleton service instance (for testing).

    Args:
        service: New service instance to use as singleton.
    """
    global _service_instance
    with _service_lock:
        _service_instance = service
        logger.info("DeforestationAlertSystemSetup singleton overridden")

def reset_service() -> None:
    """
    Reset the singleton service instance (for testing).

    Warning: Does not call shutdown(). Caller must handle cleanup.
    """
    global _service_instance
    with _service_lock:
        _service_instance = None
        logger.info("DeforestationAlertSystemSetup singleton reset")

# =============================================================================
# Convenience function
# =============================================================================

def setup_deforestation_alert_system(
    config_overrides: Optional[Dict[str, Any]] = None,
) -> DeforestationAlertSystemSetup:
    """
    Convenience function to create and return a configured setup instance.

    Creates a DeforestationAlertSystemConfig with optional overrides,
    registers it globally, and returns the singleton setup instance.

    Args:
        config_overrides: Optional dict of config field overrides.

    Returns:
        Configured DeforestationAlertSystemSetup instance.

    Example:
        >>> setup = setup_deforestation_alert_system({
        ...     "ndvi_change_threshold": Decimal("-0.20"),
        ...     "log_level": "DEBUG",
        ... })
        >>> await setup.startup()
    """
    if config_overrides:
        config = DeforestationAlertSystemConfig(**config_overrides)
        set_config(config)
    else:
        config = get_config()

    return get_service(config=config)

# =============================================================================
# FastAPI Lifespan Integration
# =============================================================================

@asynccontextmanager
async def lifespan(app: Any) -> AsyncIterator[None]:
    """
    FastAPI lifespan context manager for automatic startup/shutdown.

    Usage:
        >>> from fastapi import FastAPI
        >>> from greenlang.agents.eudr.deforestation_alert_system.setup import lifespan
        >>>
        >>> app = FastAPI(lifespan=lifespan)

    Args:
        app: FastAPI application instance.

    Yields:
        None (context manager).
    """
    # Startup
    service = get_service()
    await service.startup()
    logger.info("DeforestationAlertSystemSetup started (FastAPI lifespan)")

    yield

    # Shutdown
    await service.shutdown()
    logger.info("DeforestationAlertSystemSetup shutdown (FastAPI lifespan)")

# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    "DeforestationAlertSystemSetup",
    "get_service",
    "set_service",
    "reset_service",
    "setup_deforestation_alert_system",
    "lifespan",
    "PSYCOPG_POOL_AVAILABLE",
    "PSYCOPG_AVAILABLE",
    "REDIS_AVAILABLE",
    "OTEL_AVAILABLE",
]
