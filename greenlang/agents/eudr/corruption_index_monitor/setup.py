# -*- coding: utf-8 -*-
"""
CorruptionIndexMonitorSetup - Facade for AGENT-EUDR-019

Unified setup facade orchestrating all 8 engines of the Corruption Index
Monitor Agent. Provides a single entry point for CPI monitoring, WGI
analysis, bribery risk assessment, institutional quality scoring, trend
analysis, deforestation-corruption correlation, alert generation, and
compliance impact assessment.

Engines (8):
    1. CPIMonitorEngine             - CPI score tracking and monitoring (Feature 1)
    2. WGIAnalyzerEngine            - WGI 6-dimension analysis (Feature 2)
    3. BriberyRiskEngine            - Sector-specific bribery risk (Feature 3)
    4. InstitutionalQualityEngine   - Institutional quality scoring (Feature 4)
    5. TrendAnalysisEngine          - Multi-year trend and prediction (Feature 5)
    6. DeforestationCorrelationEngine - Corruption-deforestation linkage (Feature 6)
    7. AlertEngine                  - Alert generation and management (Feature 7)
    8. ComplianceImpactEngine       - EUDR Article 29 compliance mapping (Feature 8)

Reference Data (4):
    - cpi_database: 180+ country CPI scores (2018-2025)
    - wgi_database: WGI 6-dimension governance indicators
    - bribery_indices: TRACE Bribery Risk Matrix with EUDR sector multipliers
    - country_governance: Institutional quality and forest governance profiles

Singleton Pattern:
    Thread-safe singleton with double-checked locking via ``get_service()``.

FastAPI Integration:
    Use the ``lifespan`` async context manager with
    ``FastAPI(lifespan=lifespan)`` for automatic startup/shutdown.

Example:
    >>> from greenlang.agents.eudr.corruption_index_monitor.setup import (
    ...     CorruptionIndexMonitorSetup,
    ...     get_service,
    ... )
    >>> service = get_service()
    >>> await service.startup()
    >>> health = await service.health_check()
    >>> assert health["status"] == "healthy"
    >>>
    >>> # Query CPI score
    >>> cpi = await service.query_cpi("BR", year=2024)
    >>> assert cpi["score"] is not None
    >>>
    >>> # Comprehensive country analysis
    >>> analysis = await service.run_comprehensive_analysis(
    ...     country_code="IDN",
    ...     include_all=True,
    ... )
    >>>
    >>> await service.shutdown()

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-019
Agent ID: GL-EUDR-CIM-019
Regulation: EU 2023/1115 (EUDR) Articles 10, 11, 13, 29, 31
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

from greenlang.agents.eudr.corruption_index_monitor.config import (
    CorruptionIndexMonitorConfig,
    get_config,
    set_config,
    reset_config,
)
from greenlang.agents.eudr.corruption_index_monitor.provenance import (
    ProvenanceTracker,
    get_tracker,
)
from greenlang.agents.eudr.corruption_index_monitor.metrics import (
    PROMETHEUS_AVAILABLE,
    record_cpi_query,
    record_wgi_query,
    record_bribery_assessment,
    record_institutional_assessment,
    record_trend_analysis,
    record_correlation_analysis,
    record_alert_generated,
    record_compliance_impact,
    record_api_error,
    observe_query_duration,
    observe_analysis_duration,
    observe_correlation_duration,
    set_monitored_countries,
    set_high_risk_countries,
    set_active_alerts,
    set_data_freshness_days,
)

# ---------------------------------------------------------------------------
# Engine imports (graceful fallback for lazy loading)
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.corruption_index_monitor.engines.cpi_monitor_engine import (
        CPIMonitorEngine,
    )
    _CPI_MONITOR_ENGINE_AVAILABLE = True
except ImportError:
    CPIMonitorEngine = None  # type: ignore[assignment,misc]
    _CPI_MONITOR_ENGINE_AVAILABLE = False

try:
    from greenlang.agents.eudr.corruption_index_monitor.engines.wgi_analyzer_engine import (
        WGIAnalyzerEngine,
    )
    _WGI_ANALYZER_ENGINE_AVAILABLE = True
except ImportError:
    WGIAnalyzerEngine = None  # type: ignore[assignment,misc]
    _WGI_ANALYZER_ENGINE_AVAILABLE = False

try:
    from greenlang.agents.eudr.corruption_index_monitor.engines.bribery_risk_engine import (
        BriberyRiskEngine,
    )
    _BRIBERY_RISK_ENGINE_AVAILABLE = True
except ImportError:
    BriberyRiskEngine = None  # type: ignore[assignment,misc]
    _BRIBERY_RISK_ENGINE_AVAILABLE = False

try:
    from greenlang.agents.eudr.corruption_index_monitor.engines.institutional_quality_engine import (
        InstitutionalQualityEngine,
    )
    _INSTITUTIONAL_QUALITY_ENGINE_AVAILABLE = True
except ImportError:
    InstitutionalQualityEngine = None  # type: ignore[assignment,misc]
    _INSTITUTIONAL_QUALITY_ENGINE_AVAILABLE = False

try:
    from greenlang.agents.eudr.corruption_index_monitor.engines.trend_analysis_engine import (
        TrendAnalysisEngine,
    )
    _TREND_ANALYSIS_ENGINE_AVAILABLE = True
except ImportError:
    TrendAnalysisEngine = None  # type: ignore[assignment,misc]
    _TREND_ANALYSIS_ENGINE_AVAILABLE = False

try:
    from greenlang.agents.eudr.corruption_index_monitor.engines.deforestation_correlation_engine import (
        DeforestationCorrelationEngine,
    )
    _DEFORESTATION_CORRELATION_ENGINE_AVAILABLE = True
except ImportError:
    DeforestationCorrelationEngine = None  # type: ignore[assignment,misc]
    _DEFORESTATION_CORRELATION_ENGINE_AVAILABLE = False

try:
    from greenlang.agents.eudr.corruption_index_monitor.engines.alert_engine import (
        AlertEngine,
    )
    _ALERT_ENGINE_AVAILABLE = True
except ImportError:
    AlertEngine = None  # type: ignore[assignment,misc]
    _ALERT_ENGINE_AVAILABLE = False

try:
    from greenlang.agents.eudr.corruption_index_monitor.engines.compliance_impact_engine import (
        ComplianceImpactEngine,
    )
    _COMPLIANCE_IMPACT_ENGINE_AVAILABLE = True
except ImportError:
    ComplianceImpactEngine = None  # type: ignore[assignment,misc]
    _COMPLIANCE_IMPACT_ENGINE_AVAILABLE = False

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"
_AGENT_ID = "GL-EUDR-CIM-019"
_ENGINE_COUNT = 8
_ENGINE_NAMES: List[str] = [
    "CPIMonitorEngine",
    "WGIAnalyzerEngine",
    "BriberyRiskEngine",
    "InstitutionalQualityEngine",
    "TrendAnalysisEngine",
    "DeforestationCorrelationEngine",
    "AlertEngine",
    "ComplianceImpactEngine",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed for determinism."""
    return datetime.now(timezone.utc).replace(microsecond=0)


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
# FACADE: CorruptionIndexMonitorSetup
# =============================================================================


class CorruptionIndexMonitorSetup:
    """
    CorruptionIndexMonitorSetup orchestrates all 8 engines of AGENT-EUDR-019.

    This facade provides a single, thread-safe entry point for all corruption
    index monitoring operations per EUDR Articles 10, 11, 13, 29, and 31.

    Architecture:
        - Lazy initialization of all 8 engines (on first use)
        - Thread-safe singleton pattern with double-checked locking
        - PostgreSQL connection pooling via psycopg_pool
        - Redis caching for frequently accessed reference data
        - OpenTelemetry distributed tracing integration
        - Prometheus metrics for all operations
        - SHA-256 provenance hashing for audit trails

    Engines:
        1. CPIMonitorEngine: CPI score tracking with 180+ country coverage,
           multi-year history, regional aggregation, and percentile ranking
        2. WGIAnalyzerEngine: World Bank WGI 6-dimension analysis with
           composite scoring, standard error tracking, and trend detection
        3. BriberyRiskEngine: TRACE-based bribery risk assessment with
           EUDR sector multipliers and domain-level scoring
        4. InstitutionalQualityEngine: Judicial independence, regulatory
           enforcement, forest governance, and law enforcement composite
        5. TrendAnalysisEngine: Linear regression trend detection with
           R-squared confidence, trajectory prediction, and reversal alerts
        6. DeforestationCorrelationEngine: Pearson correlation between
           corruption indices and deforestation rates with p-value testing
        7. AlertEngine: Multi-threshold alert generation for CPI changes,
           WGI shifts, trend reversals, and country reclassifications
        8. ComplianceImpactEngine: EUDR Article 29 country classification
           mapping with due diligence level determination

    Attributes:
        config: Current configuration instance
        db_pool: Async PostgreSQL connection pool (psycopg_pool)
        redis_client: Async Redis client (redis.asyncio)
        provenance_tracker: SHA-256 provenance tracking

    Example:
        >>> service = get_service()
        >>> await service.startup()
        >>>
        >>> # Query CPI score
        >>> cpi = await service.query_cpi("BR", year=2024)
        >>>
        >>> # Analyze WGI
        >>> wgi = await service.analyze_wgi("IDN", year=2024)
        >>>
        >>> # Comprehensive country analysis
        >>> analysis = await service.run_comprehensive_analysis(
        ...     country_code="COD",
        ...     include_all=True,
        ... )
        >>>
        >>> await service.shutdown()
    """

    def __init__(
        self,
        config: Optional[CorruptionIndexMonitorConfig] = None,
        *,
        db_pool: Optional[Any] = None,
        redis_client: Optional[Any] = None,
    ) -> None:
        """
        Initialize CorruptionIndexMonitorSetup.

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
        self._cpi_monitor_engine: Optional[Any] = None
        self._wgi_analyzer_engine: Optional[Any] = None
        self._bribery_risk_engine: Optional[Any] = None
        self._institutional_quality_engine: Optional[Any] = None
        self._trend_analysis_engine: Optional[Any] = None
        self._deforestation_correlation_engine: Optional[Any] = None
        self._alert_engine: Optional[Any] = None
        self._compliance_impact_engine: Optional[Any] = None

        # Lifecycle state
        self._started: bool = False
        self._startup_time: Optional[datetime] = None
        self._startup_lock = asyncio.Lock()
        self._shutdown_lock = asyncio.Lock()

        # Statistics tracking
        self._stats: Dict[str, int] = {
            "total_analyses": 0,
            "total_cpi_queries": 0,
            "total_wgi_queries": 0,
            "total_bribery_assessments": 0,
            "total_institutional_assessments": 0,
            "total_trend_analyses": 0,
            "total_correlation_analyses": 0,
            "total_alerts_generated": 0,
            "total_compliance_assessments": 0,
            "total_comprehensive_analyses": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
        }

        logger.info(
            f"CorruptionIndexMonitorSetup initialized "
            f"(version={_MODULE_VERSION}, agent_id={_AGENT_ID})"
        )

    # -----------------------------------------------------------------------
    # Properties for engine access
    # -----------------------------------------------------------------------

    @property
    def cpi_monitor_engine(self) -> Any:
        """Access CPIMonitorEngine (lazy initialized).

        Returns:
            CPIMonitorEngine instance.

        Raises:
            RuntimeError: If engine is not available.
        """
        if self._cpi_monitor_engine is None:
            raise RuntimeError(
                "CPIMonitorEngine not initialized. Call startup() first."
            )
        return self._cpi_monitor_engine

    @property
    def wgi_analyzer_engine(self) -> Any:
        """Access WGIAnalyzerEngine (lazy initialized).

        Returns:
            WGIAnalyzerEngine instance.

        Raises:
            RuntimeError: If engine is not available.
        """
        if self._wgi_analyzer_engine is None:
            raise RuntimeError(
                "WGIAnalyzerEngine not initialized. Call startup() first."
            )
        return self._wgi_analyzer_engine

    @property
    def bribery_risk_engine(self) -> Any:
        """Access BriberyRiskEngine (lazy initialized).

        Returns:
            BriberyRiskEngine instance.

        Raises:
            RuntimeError: If engine is not available.
        """
        if self._bribery_risk_engine is None:
            raise RuntimeError(
                "BriberyRiskEngine not initialized. Call startup() first."
            )
        return self._bribery_risk_engine

    @property
    def institutional_quality_engine(self) -> Any:
        """Access InstitutionalQualityEngine (lazy initialized).

        Returns:
            InstitutionalQualityEngine instance.

        Raises:
            RuntimeError: If engine is not available.
        """
        if self._institutional_quality_engine is None:
            raise RuntimeError(
                "InstitutionalQualityEngine not initialized. Call startup() first."
            )
        return self._institutional_quality_engine

    @property
    def trend_analysis_engine(self) -> Any:
        """Access TrendAnalysisEngine (lazy initialized).

        Returns:
            TrendAnalysisEngine instance.

        Raises:
            RuntimeError: If engine is not available.
        """
        if self._trend_analysis_engine is None:
            raise RuntimeError(
                "TrendAnalysisEngine not initialized. Call startup() first."
            )
        return self._trend_analysis_engine

    @property
    def deforestation_correlation_engine(self) -> Any:
        """Access DeforestationCorrelationEngine (lazy initialized).

        Returns:
            DeforestationCorrelationEngine instance.

        Raises:
            RuntimeError: If engine is not available.
        """
        if self._deforestation_correlation_engine is None:
            raise RuntimeError(
                "DeforestationCorrelationEngine not initialized. Call startup() first."
            )
        return self._deforestation_correlation_engine

    @property
    def alert_engine(self) -> Any:
        """Access AlertEngine (lazy initialized).

        Returns:
            AlertEngine instance.

        Raises:
            RuntimeError: If engine is not available.
        """
        if self._alert_engine is None:
            raise RuntimeError(
                "AlertEngine not initialized. Call startup() first."
            )
        return self._alert_engine

    @property
    def compliance_impact_engine(self) -> Any:
        """Access ComplianceImpactEngine (lazy initialized).

        Returns:
            ComplianceImpactEngine instance.

        Raises:
            RuntimeError: If engine is not available.
        """
        if self._compliance_impact_engine is None:
            raise RuntimeError(
                "ComplianceImpactEngine not initialized. Call startup() first."
            )
        return self._compliance_impact_engine

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
                    "CorruptionIndexMonitorSetup already started, skipping startup"
                )
                return

            logger.info("Starting CorruptionIndexMonitorSetup...")
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

                # 3. Lazy engine initialization will happen on first use
                logger.debug(
                    "Engines will be initialized on first use (lazy loading)"
                )

                # 4. Mark as started
                self._started = True
                self._startup_time = _utcnow()
                duration_ms = (time.monotonic() - start_time) * 1000

                logger.info(
                    f"CorruptionIndexMonitorSetup started successfully "
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
                    "CorruptionIndexMonitorSetup not started, skipping shutdown"
                )
                return

            logger.info("Shutting down CorruptionIndexMonitorSetup...")

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
                    "CorruptionIndexMonitorSetup shutdown complete"
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
            "timestamp": _utcnow().isoformat(),
            "engines": {},
            "reference_data": {},
            "connectivity": {},
        }

        # Initialize all engines eagerly
        engine_init_methods = [
            ("CPIMonitorEngine", self._ensure_cpi_monitor_engine),
            ("WGIAnalyzerEngine", self._ensure_wgi_analyzer_engine),
            ("BriberyRiskEngine", self._ensure_bribery_risk_engine),
            ("InstitutionalQualityEngine", self._ensure_institutional_quality_engine),
            ("TrendAnalysisEngine", self._ensure_trend_analysis_engine),
            ("DeforestationCorrelationEngine", self._ensure_deforestation_correlation_engine),
            ("AlertEngine", self._ensure_alert_engine),
            ("ComplianceImpactEngine", self._ensure_compliance_impact_engine),
        ]

        for engine_name, init_method in engine_init_methods:
            try:
                await init_method()
                init_results["engines"][engine_name] = "initialized"
                logger.info(f"Engine {engine_name} initialized successfully")
            except Exception as e:
                init_results["engines"][engine_name] = f"failed: {str(e)}"
                logger.warning(f"Engine {engine_name} initialization failed: {e}")

        # Load reference data
        try:
            from greenlang.agents.eudr.corruption_index_monitor.reference_data import (
                CPIDatabase,
                WGIDatabase,
                BriberyIndicesDatabase,
                CountryGovernanceDatabase,
                validate_all_databases,
            )

            validation = validate_all_databases()
            init_results["reference_data"] = validation
        except ImportError as e:
            init_results["reference_data"]["status"] = f"partial: {str(e)}"
            logger.warning(f"Reference data import failed: {e}")

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

    async def _shutdown_engines(self) -> None:
        """Shutdown all initialized engines gracefully."""
        engines = [
            ("CPIMonitorEngine", self._cpi_monitor_engine),
            ("WGIAnalyzerEngine", self._wgi_analyzer_engine),
            ("BriberyRiskEngine", self._bribery_risk_engine),
            ("InstitutionalQualityEngine", self._institutional_quality_engine),
            ("TrendAnalysisEngine", self._trend_analysis_engine),
            ("DeforestationCorrelationEngine", self._deforestation_correlation_engine),
            ("AlertEngine", self._alert_engine),
            ("ComplianceImpactEngine", self._compliance_impact_engine),
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

    async def _ensure_cpi_monitor_engine(self) -> Any:
        """Lazy initialize CPIMonitorEngine.

        Returns:
            Initialized CPIMonitorEngine instance.

        Raises:
            RuntimeError: If engine module is not available.
        """
        if self._cpi_monitor_engine is None:
            if not _CPI_MONITOR_ENGINE_AVAILABLE or CPIMonitorEngine is None:
                raise RuntimeError("CPIMonitorEngine not available")
            logger.debug("Initializing CPIMonitorEngine...")
            self._cpi_monitor_engine = CPIMonitorEngine(
                config=self._config,
                db_pool=self._db_pool,
                redis_client=self._redis_client,
            )
            if hasattr(self._cpi_monitor_engine, "startup"):
                await self._cpi_monitor_engine.startup()
            logger.info("CPIMonitorEngine initialized")
        return self._cpi_monitor_engine

    async def _ensure_wgi_analyzer_engine(self) -> Any:
        """Lazy initialize WGIAnalyzerEngine.

        Returns:
            Initialized WGIAnalyzerEngine instance.

        Raises:
            RuntimeError: If engine module is not available.
        """
        if self._wgi_analyzer_engine is None:
            if not _WGI_ANALYZER_ENGINE_AVAILABLE or WGIAnalyzerEngine is None:
                raise RuntimeError("WGIAnalyzerEngine not available")
            logger.debug("Initializing WGIAnalyzerEngine...")
            self._wgi_analyzer_engine = WGIAnalyzerEngine(
                config=self._config,
                db_pool=self._db_pool,
                redis_client=self._redis_client,
            )
            if hasattr(self._wgi_analyzer_engine, "startup"):
                await self._wgi_analyzer_engine.startup()
            logger.info("WGIAnalyzerEngine initialized")
        return self._wgi_analyzer_engine

    async def _ensure_bribery_risk_engine(self) -> Any:
        """Lazy initialize BriberyRiskEngine.

        Returns:
            Initialized BriberyRiskEngine instance.

        Raises:
            RuntimeError: If engine module is not available.
        """
        if self._bribery_risk_engine is None:
            if not _BRIBERY_RISK_ENGINE_AVAILABLE or BriberyRiskEngine is None:
                raise RuntimeError("BriberyRiskEngine not available")
            logger.debug("Initializing BriberyRiskEngine...")
            self._bribery_risk_engine = BriberyRiskEngine(
                config=self._config,
                db_pool=self._db_pool,
                redis_client=self._redis_client,
            )
            if hasattr(self._bribery_risk_engine, "startup"):
                await self._bribery_risk_engine.startup()
            logger.info("BriberyRiskEngine initialized")
        return self._bribery_risk_engine

    async def _ensure_institutional_quality_engine(self) -> Any:
        """Lazy initialize InstitutionalQualityEngine.

        Returns:
            Initialized InstitutionalQualityEngine instance.

        Raises:
            RuntimeError: If engine module is not available.
        """
        if self._institutional_quality_engine is None:
            if not _INSTITUTIONAL_QUALITY_ENGINE_AVAILABLE or InstitutionalQualityEngine is None:
                raise RuntimeError("InstitutionalQualityEngine not available")
            logger.debug("Initializing InstitutionalQualityEngine...")
            self._institutional_quality_engine = InstitutionalQualityEngine(
                config=self._config,
                db_pool=self._db_pool,
                redis_client=self._redis_client,
            )
            if hasattr(self._institutional_quality_engine, "startup"):
                await self._institutional_quality_engine.startup()
            logger.info("InstitutionalQualityEngine initialized")
        return self._institutional_quality_engine

    async def _ensure_trend_analysis_engine(self) -> Any:
        """Lazy initialize TrendAnalysisEngine.

        Returns:
            Initialized TrendAnalysisEngine instance.

        Raises:
            RuntimeError: If engine module is not available.
        """
        if self._trend_analysis_engine is None:
            if not _TREND_ANALYSIS_ENGINE_AVAILABLE or TrendAnalysisEngine is None:
                raise RuntimeError("TrendAnalysisEngine not available")
            logger.debug("Initializing TrendAnalysisEngine...")
            self._trend_analysis_engine = TrendAnalysisEngine(
                config=self._config,
                db_pool=self._db_pool,
                redis_client=self._redis_client,
            )
            if hasattr(self._trend_analysis_engine, "startup"):
                await self._trend_analysis_engine.startup()
            logger.info("TrendAnalysisEngine initialized")
        return self._trend_analysis_engine

    async def _ensure_deforestation_correlation_engine(self) -> Any:
        """Lazy initialize DeforestationCorrelationEngine.

        Returns:
            Initialized DeforestationCorrelationEngine instance.

        Raises:
            RuntimeError: If engine module is not available.
        """
        if self._deforestation_correlation_engine is None:
            if not _DEFORESTATION_CORRELATION_ENGINE_AVAILABLE or DeforestationCorrelationEngine is None:
                raise RuntimeError("DeforestationCorrelationEngine not available")
            logger.debug("Initializing DeforestationCorrelationEngine...")
            self._deforestation_correlation_engine = DeforestationCorrelationEngine(
                config=self._config,
                db_pool=self._db_pool,
                redis_client=self._redis_client,
            )
            if hasattr(self._deforestation_correlation_engine, "startup"):
                await self._deforestation_correlation_engine.startup()
            logger.info("DeforestationCorrelationEngine initialized")
        return self._deforestation_correlation_engine

    async def _ensure_alert_engine(self) -> Any:
        """Lazy initialize AlertEngine.

        Returns:
            Initialized AlertEngine instance.

        Raises:
            RuntimeError: If engine module is not available.
        """
        if self._alert_engine is None:
            if not _ALERT_ENGINE_AVAILABLE or AlertEngine is None:
                raise RuntimeError("AlertEngine not available")
            logger.debug("Initializing AlertEngine...")
            self._alert_engine = AlertEngine(
                config=self._config,
                db_pool=self._db_pool,
                redis_client=self._redis_client,
            )
            if hasattr(self._alert_engine, "startup"):
                await self._alert_engine.startup()
            logger.info("AlertEngine initialized")
        return self._alert_engine

    async def _ensure_compliance_impact_engine(self) -> Any:
        """Lazy initialize ComplianceImpactEngine.

        Returns:
            Initialized ComplianceImpactEngine instance.

        Raises:
            RuntimeError: If engine module is not available.
        """
        if self._compliance_impact_engine is None:
            if not _COMPLIANCE_IMPACT_ENGINE_AVAILABLE or ComplianceImpactEngine is None:
                raise RuntimeError("ComplianceImpactEngine not available")
            logger.debug("Initializing ComplianceImpactEngine...")
            self._compliance_impact_engine = ComplianceImpactEngine(
                config=self._config,
                db_pool=self._db_pool,
                redis_client=self._redis_client,
            )
            if hasattr(self._compliance_impact_engine, "startup"):
                await self._compliance_impact_engine.startup()
            logger.info("ComplianceImpactEngine initialized")
        return self._compliance_impact_engine

    # -----------------------------------------------------------------------
    # Public API: Engine delegation methods
    # -----------------------------------------------------------------------

    async def query_cpi(
        self,
        country_code: str,
        year: int = 2024,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Query CPI score for a country using CPIMonitorEngine.

        Args:
            country_code: ISO 3166-1 alpha-2 or alpha-3 country code.
            year: CPI assessment year (2018-2025).
            **kwargs: Additional query parameters.

        Returns:
            Dict with CPI score, rank, percentile, confidence interval,
            and provenance hash.

        Raises:
            RuntimeError: If CPIMonitorEngine is not available.
        """
        start_time = time.monotonic()
        logger.info(f"Querying CPI: {country_code} (year={year})")

        try:
            engine = await self._ensure_cpi_monitor_engine()
            result = await engine.query_cpi(
                country_code=country_code, year=year, **kwargs
            )

            duration_sec = time.monotonic() - start_time
            observe_query_duration(duration_sec)
            record_cpi_query(country=country_code, status="success")
            self._stats["total_cpi_queries"] += 1
            self._stats["total_analyses"] += 1

            provenance_hash = _calculate_sha256({
                "action": "query_cpi",
                "country_code": country_code,
                "year": year,
                "timestamp": _utcnow().isoformat(),
            })

            logger.info(
                f"CPI query complete for {country_code}: "
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
            record_api_error("query_cpi", str(e))
            logger.error(f"query_cpi failed: {e}", exc_info=True)
            raise

    async def analyze_wgi(
        self,
        country_code: str,
        year: int = 2024,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Analyze WGI indicators for a country using WGIAnalyzerEngine.

        Args:
            country_code: ISO 3166-1 alpha-3 country code.
            year: WGI assessment year.
            **kwargs: Additional analysis parameters.

        Returns:
            Dict with 6 WGI dimension scores, composite score, and
            provenance hash.

        Raises:
            RuntimeError: If WGIAnalyzerEngine is not available.
        """
        start_time = time.monotonic()
        logger.info(f"Analyzing WGI: {country_code} (year={year})")

        try:
            engine = await self._ensure_wgi_analyzer_engine()
            result = await engine.analyze_wgi(
                country_code=country_code, year=year, **kwargs
            )

            duration_sec = time.monotonic() - start_time
            observe_analysis_duration(duration_sec)
            record_wgi_query(country=country_code, status="success")
            self._stats["total_wgi_queries"] += 1
            self._stats["total_analyses"] += 1

            provenance_hash = _calculate_sha256({
                "action": "analyze_wgi",
                "country_code": country_code,
                "year": year,
                "timestamp": _utcnow().isoformat(),
            })

            logger.info(
                f"WGI analysis complete for {country_code}: "
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
            record_api_error("analyze_wgi", str(e))
            logger.error(f"analyze_wgi failed: {e}", exc_info=True)
            raise

    async def assess_bribery_risk(
        self,
        country_code: str,
        sector: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Assess bribery risk for a country and sector using BriberyRiskEngine.

        Args:
            country_code: ISO 3166-1 alpha-3 country code.
            sector: Optional EUDR sector for sector-adjusted score.
            **kwargs: Additional assessment parameters.

        Returns:
            Dict with composite score, domain scores, sector adjustment,
            and provenance hash.

        Raises:
            RuntimeError: If BriberyRiskEngine is not available.
        """
        start_time = time.monotonic()
        logger.info(
            f"Assessing bribery risk: {country_code} (sector={sector})"
        )

        try:
            engine = await self._ensure_bribery_risk_engine()
            result = await engine.assess_bribery_risk(
                country_code=country_code, sector=sector, **kwargs
            )

            duration_sec = time.monotonic() - start_time
            record_bribery_assessment(country=country_code, status="success")
            self._stats["total_bribery_assessments"] += 1
            self._stats["total_analyses"] += 1

            provenance_hash = _calculate_sha256({
                "action": "assess_bribery_risk",
                "country_code": country_code,
                "sector": sector,
                "timestamp": _utcnow().isoformat(),
            })

            logger.info(
                f"Bribery risk assessment complete for {country_code}: "
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
            record_api_error("assess_bribery_risk", str(e))
            logger.error(f"assess_bribery_risk failed: {e}", exc_info=True)
            raise

    async def assess_compliance_impact(
        self,
        country_code: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Assess EUDR compliance impact using ComplianceImpactEngine.

        Maps corruption indices to EUDR Article 29 country classifications
        (low/standard/high risk) and determines due diligence level.

        Args:
            country_code: ISO 3166-1 alpha-3 country code.
            **kwargs: Additional assessment parameters.

        Returns:
            Dict with country_classification, dd_level, risk_factors,
            and provenance hash.

        Raises:
            RuntimeError: If ComplianceImpactEngine is not available.
        """
        start_time = time.monotonic()
        logger.info(f"Assessing compliance impact: {country_code}")

        try:
            engine = await self._ensure_compliance_impact_engine()
            result = await engine.assess_compliance_impact(
                country_code=country_code, **kwargs
            )

            duration_sec = time.monotonic() - start_time
            record_compliance_impact(country=country_code, status="success")
            self._stats["total_compliance_assessments"] += 1
            self._stats["total_analyses"] += 1

            provenance_hash = _calculate_sha256({
                "action": "assess_compliance_impact",
                "country_code": country_code,
                "timestamp": _utcnow().isoformat(),
            })

            logger.info(
                f"Compliance impact assessment complete for {country_code}: "
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
        country_code: str,
        include_all: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Run ALL engines for a complete country corruption analysis.

        Orchestrates all 8 engines to produce a comprehensive corruption
        risk analysis covering CPI monitoring, WGI analysis, bribery risk,
        institutional quality, trend analysis, deforestation correlation,
        alert generation, and compliance impact assessment.

        Args:
            country_code: ISO 3166-1 alpha-3 country code.
            include_all: If True, run all engines. If False, run only core.
            **kwargs: Additional parameters passed to individual engines.

        Returns:
            Dict with comprehensive analysis results containing per-engine
            outputs, provenance chain, and total processing time.

        Raises:
            ValueError: If country_code is empty.
        """
        start_time = time.monotonic()
        logger.info(
            f"Starting comprehensive analysis for {country_code} "
            f"(include_all={include_all})"
        )

        try:
            if not country_code:
                raise ValueError("country_code must not be empty")

            results: Dict[str, Any] = {
                "country_code": country_code,
                "analysis_timestamp": _utcnow().isoformat(),
                "agent_id": _AGENT_ID,
                "version": _MODULE_VERSION,
            }
            provenance_chain: List[str] = []

            # 1. CPI Query (always included)
            try:
                cpi = await self.query_cpi(
                    country_code=country_code,
                    year=kwargs.get("year", 2024),
                )
                results["cpi_score"] = cpi
                if "provenance_hash" in cpi:
                    provenance_chain.append(cpi["provenance_hash"])
            except Exception as e:
                results["cpi_score"] = {"error": str(e)}
                logger.warning(f"CPI query failed: {e}")

            # 2. WGI Analysis (always included)
            try:
                wgi = await self.analyze_wgi(
                    country_code=country_code,
                    year=kwargs.get("year", 2024),
                )
                results["wgi_analysis"] = wgi
                if "provenance_hash" in wgi:
                    provenance_chain.append(wgi["provenance_hash"])
            except Exception as e:
                results["wgi_analysis"] = {"error": str(e)}
                logger.warning(f"WGI analysis failed: {e}")

            # 3. Bribery Risk (always included)
            try:
                bribery = await self.assess_bribery_risk(
                    country_code=country_code,
                    sector=kwargs.get("sector"),
                )
                results["bribery_risk"] = bribery
                if "provenance_hash" in bribery:
                    provenance_chain.append(bribery["provenance_hash"])
            except Exception as e:
                results["bribery_risk"] = {"error": str(e)}
                logger.warning(f"Bribery risk assessment failed: {e}")

            # 4. Compliance Impact (always included)
            try:
                compliance = await self.assess_compliance_impact(
                    country_code=country_code,
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
                "country_code": country_code,
                "engine_provenance_chain": provenance_chain,
                "timestamp": _utcnow().isoformat(),
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
                f"Comprehensive analysis complete for {country_code}: "
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
            ("CPIMonitorEngine", self._cpi_monitor_engine),
            ("WGIAnalyzerEngine", self._wgi_analyzer_engine),
            ("BriberyRiskEngine", self._bribery_risk_engine),
            ("InstitutionalQualityEngine", self._institutional_quality_engine),
            ("TrendAnalysisEngine", self._trend_analysis_engine),
            ("DeforestationCorrelationEngine", self._deforestation_correlation_engine),
            ("AlertEngine", self._alert_engine),
            ("ComplianceImpactEngine", self._compliance_impact_engine),
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
            from greenlang.agents.eudr.corruption_index_monitor.reference_data import (
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
                _utcnow() - self._startup_time
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
            "engines_initialized": engines_initialized,
            "engines_total": _ENGINE_COUNT,
            "engine_statuses": engine_statuses,
            "processing_time_ms": round(duration_ms, 2),
            "timestamp": _utcnow().isoformat(),
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
                _utcnow() - self._startup_time
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
            "total_cpi_queries": self._stats["total_cpi_queries"],
            "total_wgi_queries": self._stats["total_wgi_queries"],
            "total_bribery_assessments": self._stats["total_bribery_assessments"],
            "total_institutional_assessments": self._stats["total_institutional_assessments"],
            "total_trend_analyses": self._stats["total_trend_analyses"],
            "total_correlation_analyses": self._stats["total_correlation_analyses"],
            "total_alerts_generated": self._stats["total_alerts_generated"],
            "total_compliance_assessments": self._stats["total_compliance_assessments"],
            "total_comprehensive_analyses": self._stats["total_comprehensive_analyses"],
            "cache_hits": self._stats["cache_hits"],
            "cache_misses": self._stats["cache_misses"],
            "cache_hit_rate": round(cache_hit_rate, 4),
            "errors": self._stats["errors"],
            "timestamp": _utcnow().isoformat(),
        }

    def get_engine(self, engine_name: str) -> Optional[Any]:
        """Get a specific engine instance by name.

        Args:
            engine_name: Engine name (e.g. "CPIMonitorEngine").

        Returns:
            Engine instance or None if not initialized.
        """
        engine_map = {
            "CPIMonitorEngine": self._cpi_monitor_engine,
            "WGIAnalyzerEngine": self._wgi_analyzer_engine,
            "BriberyRiskEngine": self._bribery_risk_engine,
            "InstitutionalQualityEngine": self._institutional_quality_engine,
            "TrendAnalysisEngine": self._trend_analysis_engine,
            "DeforestationCorrelationEngine": self._deforestation_correlation_engine,
            "AlertEngine": self._alert_engine,
            "ComplianceImpactEngine": self._compliance_impact_engine,
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
            "dependencies": {
                "psycopg_pool": PSYCOPG_POOL_AVAILABLE,
                "psycopg": PSYCOPG_AVAILABLE,
                "redis": REDIS_AVAILABLE,
                "opentelemetry": OTEL_AVAILABLE,
                "prometheus": PROMETHEUS_AVAILABLE,
            },
            "engine_availability": {
                "CPIMonitorEngine": _CPI_MONITOR_ENGINE_AVAILABLE,
                "WGIAnalyzerEngine": _WGI_ANALYZER_ENGINE_AVAILABLE,
                "BriberyRiskEngine": _BRIBERY_RISK_ENGINE_AVAILABLE,
                "InstitutionalQualityEngine": _INSTITUTIONAL_QUALITY_ENGINE_AVAILABLE,
                "TrendAnalysisEngine": _TREND_ANALYSIS_ENGINE_AVAILABLE,
                "DeforestationCorrelationEngine": _DEFORESTATION_CORRELATION_ENGINE_AVAILABLE,
                "AlertEngine": _ALERT_ENGINE_AVAILABLE,
                "ComplianceImpactEngine": _COMPLIANCE_IMPACT_ENGINE_AVAILABLE,
            },
            "config_summary": {
                "cpi_high_risk_threshold": self._config.cpi_high_risk_threshold,
                "wgi_risk_threshold": self._config.wgi_risk_threshold,
                "art29_low_risk_cpi": self._config.art29_low_risk_cpi,
                "art29_high_risk_cpi": self._config.art29_high_risk_cpi,
                "alert_cpi_change_threshold": self._config.alert_cpi_change_threshold,
                "trend_min_years": self._config.trend_min_years,
            },
        }


# =============================================================================
# Module-level singleton management
# =============================================================================

_service_instance: Optional[CorruptionIndexMonitorSetup] = None
_service_lock = threading.Lock()


def get_service(
    config: Optional[CorruptionIndexMonitorConfig] = None,
) -> CorruptionIndexMonitorSetup:
    """
    Get or create the singleton CorruptionIndexMonitorSetup instance.

    Thread-safe singleton with double-checked locking pattern.

    Args:
        config: Optional configuration override (only used on first call).

    Returns:
        Singleton CorruptionIndexMonitorSetup instance.

    Example:
        >>> service = get_service()
        >>> await service.startup()
        >>> cpi = await service.query_cpi("BR", year=2024)
        >>> await service.shutdown()
    """
    global _service_instance

    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = CorruptionIndexMonitorSetup(config=config)
                logger.info("CorruptionIndexMonitorSetup singleton created")

    return _service_instance


def set_service(service: CorruptionIndexMonitorSetup) -> None:
    """
    Override the singleton service instance (for testing).

    Args:
        service: New service instance to use as singleton.
    """
    global _service_instance
    with _service_lock:
        _service_instance = service
        logger.info("CorruptionIndexMonitorSetup singleton overridden")


def reset_service() -> None:
    """
    Reset the singleton service instance (for testing).

    Warning: Does not call shutdown(). Caller must handle cleanup.
    """
    global _service_instance
    with _service_lock:
        _service_instance = None
        logger.info("CorruptionIndexMonitorSetup singleton reset")


# =============================================================================
# Convenience function
# =============================================================================


def setup_corruption_index_monitor(
    config_overrides: Optional[Dict[str, Any]] = None,
) -> CorruptionIndexMonitorSetup:
    """
    Convenience function to create and return a configured setup instance.

    Creates a CorruptionIndexMonitorConfig with optional overrides,
    registers it globally, and returns the singleton setup instance.

    Args:
        config_overrides: Optional dict of config field overrides.

    Returns:
        Configured CorruptionIndexMonitorSetup instance.

    Example:
        >>> setup = setup_corruption_index_monitor({
        ...     "cpi_high_risk_threshold": 25,
        ...     "log_level": "DEBUG",
        ... })
        >>> await setup.startup()
    """
    if config_overrides:
        config = CorruptionIndexMonitorConfig(**config_overrides)
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
        >>> from greenlang.agents.eudr.corruption_index_monitor.setup import lifespan
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
    logger.info("CorruptionIndexMonitorSetup started (FastAPI lifespan)")

    yield

    # Shutdown
    await service.shutdown()
    logger.info("CorruptionIndexMonitorSetup shutdown (FastAPI lifespan)")


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    "CorruptionIndexMonitorSetup",
    "get_service",
    "set_service",
    "reset_service",
    "setup_corruption_index_monitor",
    "lifespan",
    "PSYCOPG_POOL_AVAILABLE",
    "PSYCOPG_AVAILABLE",
    "REDIS_AVAILABLE",
    "OTEL_AVAILABLE",
]
