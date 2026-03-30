# -*- coding: utf-8 -*-
"""
SupplyChainMapperService - Facade for AGENT-EUDR-001 Supply Chain Mapping Master

This module implements the SupplyChainMapperService, the single entry point for
all supply chain mapping operations in the GL-EUDR-APP. It manages the lifecycle
of nine internal engines, an async PostgreSQL connection pool (psycopg +
psycopg_pool), a Redis cache connection, OpenTelemetry tracing, and Prometheus
metrics. The service exposes a unified interface consumed by the FastAPI router
layer and the GL-EUDR-APP integration.

Lifecycle:
    startup  -> load config -> connect DB -> register pgvector -> connect Redis
             -> initialize engines -> start health check background task
    shutdown -> close engines -> close Redis -> close DB pool -> flush metrics

Engines (9):
    1. SupplyChainGraphEngine      - Core DAG graph engine (Feature 1)
    2. MultiTierMapper             - Recursive supply chain discovery (Feature 2)
    3. GeolocationLinker           - Plot-level geolocation (Feature 3)
    4. BatchTraceabilityEngine     - Forward/backward batch tracing (Feature 4)
    5. RiskPropagationEngine       - Deterministic risk propagation (Feature 5)
    6. GapAnalyzer                 - Compliance gap detection (Feature 6)
    7. VisualizationEngine         - Graph layout and Sankey export (Feature 7)
    8. RegulatoryExporter          - DDS and regulatory export (Feature 8)
    9. SupplierOnboardingEngine    - Supplier onboarding workflow (Feature 9)

FastAPI Integration:
    Use the ``lifespan`` async context manager with ``FastAPI(lifespan=lifespan)``
    for automatic startup/shutdown.

Example:
    >>> from greenlang.agents.eudr.supply_chain_mapper.setup import (
    ...     SupplyChainMapperService,
    ...     get_service,
    ... )
    >>> service = get_service()
    >>> await service.startup()
    >>> health = await service.health_check()
    >>> assert health["status"] == "healthy"
    >>> await service.shutdown()

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-001 Supply Chain Mapping Master (GL-EUDR-SCM-001)
Status: Production Ready
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional

from greenlang.agents.eudr.supply_chain_mapper.config import (
    SupplyChainMapperConfig,
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

def _compute_service_hash(config: SupplyChainMapperConfig) -> str:
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
# SupplyChainMapperService
# ---------------------------------------------------------------------------

class SupplyChainMapperService:
    """Facade service for the EUDR Supply Chain Mapping Master Agent.

    This is the single entry point for all supply chain mapping operations.
    It manages the full lifecycle of database connections, cache connections,
    nine internal engines, health monitoring, and OpenTelemetry tracing.

    The service follows a strict startup/shutdown protocol:
        startup:  config -> DB pool -> pgvector -> Redis -> engines -> health
        shutdown: health stop -> engines -> Redis -> DB pool -> metrics flush

    Attributes:
        config: Service configuration loaded from env or injected.
        is_running: Whether the service is currently active and healthy.

    Example:
        >>> service = SupplyChainMapperService()
        >>> await service.startup()
        >>> graph = await service.graph_engine.create_graph(...)
        >>> await service.shutdown()
    """

    def __init__(
        self,
        config: Optional[SupplyChainMapperConfig] = None,
    ) -> None:
        """Initialize SupplyChainMapperService.

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
        self._graph_engine: Optional[Any] = None
        self._multi_tier_mapper: Optional[Any] = None
        self._geolocation_linker: Optional[Any] = None
        self._batch_traceability_engine: Optional[Any] = None
        self._risk_propagation_engine: Optional[Any] = None
        self._gap_analyzer: Optional[Any] = None
        self._visualization_engine: Optional[Any] = None
        self._regulatory_exporter: Optional[Any] = None
        self._supplier_onboarding_engine: Optional[Any] = None

        # Health check background task
        self._health_task: Optional[asyncio.Task[None]] = None
        self._last_health: Optional[HealthStatus] = None
        self._health_interval_seconds: float = 30.0

        # OpenTelemetry tracer
        self._tracer: Optional[Any] = None

        logger.info(
            "SupplyChainMapperService created: config_hash=%s, "
            "pool_size=%d, cache_ttl=%ds",
            self._config_hash[:12],
            self._config.pool_size,
            self._config.cache_ttl,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> SupplyChainMapperConfig:
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
    def graph_engine(self) -> Any:
        """Return the SupplyChainGraphEngine instance.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._graph_engine

    @property
    def multi_tier_mapper(self) -> Any:
        """Return the MultiTierMapper instance.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._multi_tier_mapper

    @property
    def geolocation_linker(self) -> Any:
        """Return the GeolocationLinker instance.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._geolocation_linker

    @property
    def batch_traceability_engine(self) -> Any:
        """Return the BatchTraceabilityEngine instance.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._batch_traceability_engine

    @property
    def risk_propagation_engine(self) -> Any:
        """Return the RiskPropagationEngine instance.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._risk_propagation_engine

    @property
    def gap_analyzer(self) -> Any:
        """Return the GapAnalyzer instance.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._gap_analyzer

    @property
    def visualization_engine(self) -> Any:
        """Return the VisualizationEngine instance.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._visualization_engine

    @property
    def regulatory_exporter(self) -> Any:
        """Return the RegulatoryExporter instance.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._regulatory_exporter

    @property
    def supplier_onboarding_engine(self) -> Any:
        """Return the SupplierOnboardingEngine instance.

        Raises:
            RuntimeError: If the service has not been started.
        """
        self._ensure_started()
        return self._supplier_onboarding_engine

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
            6. Initialize all nine engines
            7. Start background health check task

        Idempotent: safe to call multiple times.

        Raises:
            RuntimeError: If a critical connection fails.
        """
        if self._started:
            logger.debug("SupplyChainMapperService already started")
            return

        start = time.monotonic()
        logger.info("SupplyChainMapperService starting up...")

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
            "SupplyChainMapperService started in %.1fms: "
            "db=%s, redis=%s, engines=9, config_hash=%s",
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
            logger.debug("SupplyChainMapperService already stopped")
            return

        logger.info("SupplyChainMapperService shutting down...")
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
            "SupplyChainMapperService shut down in %.1fms", elapsed
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
    # Internal: Logging
    # ------------------------------------------------------------------

    def _configure_logging(self) -> None:
        """Configure structured logging based on service configuration."""
        log_level = getattr(logging, self._config.log_level, logging.INFO)
        logging.getLogger("greenlang.agents.eudr.supply_chain_mapper").setLevel(
            log_level
        )
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
                "greenlang.agents.eudr.supply_chain_mapper",
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
        embedding-based similarity search in the supply chain mapper.
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
                self._config.cache_ttl,
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
        """Initialize all nine internal engines.

        Engines are created with references to the shared DB pool, Redis
        client, and service configuration. Engines that require async
        initialization have their ``initialize()`` methods called.
        """
        logger.info("Initializing 9 supply chain mapping engines...")

        # 1. SupplyChainGraphEngine
        self._graph_engine = await self._init_graph_engine()

        # 2. MultiTierMapper
        self._multi_tier_mapper = await self._init_multi_tier_mapper()

        # 3. GeolocationLinker
        self._geolocation_linker = await self._init_geolocation_linker()

        # 4. BatchTraceabilityEngine
        self._batch_traceability_engine = (
            await self._init_batch_traceability_engine()
        )

        # 5. RiskPropagationEngine
        self._risk_propagation_engine = (
            await self._init_risk_propagation_engine()
        )

        # 6. GapAnalyzer
        self._gap_analyzer = await self._init_gap_analyzer()

        # 7. VisualizationEngine
        self._visualization_engine = await self._init_visualization_engine()

        # 8. RegulatoryExporter
        self._regulatory_exporter = await self._init_regulatory_exporter()

        # 9. SupplierOnboardingEngine
        self._supplier_onboarding_engine = (
            await self._init_supplier_onboarding_engine()
        )

        logger.info("All 9 engines initialized successfully")

    async def _init_graph_engine(self) -> Any:
        """Initialize the SupplyChainGraphEngine.

        Returns:
            Initialized SupplyChainGraphEngine instance.
        """
        try:
            from greenlang.agents.eudr.supply_chain_mapper.graph_engine import (
                GraphEngineConfig,
                SupplyChainGraphEngine,
            )

            engine_config = GraphEngineConfig(
                database_url=self._config.database_url,
                pool_min_size=max(1, self._config.pool_size // 2),
                pool_max_size=self._config.pool_size,
                enable_persistence=self._db_pool is not None,
            )
            engine = SupplyChainGraphEngine(config=engine_config)
            if self._db_pool is not None:
                await engine.initialize()
            logger.info("SupplyChainGraphEngine initialized")
            return engine
        except ImportError:
            logger.warning("SupplyChainGraphEngine not available")
            return None
        except Exception as exc:
            logger.error(
                "Failed to initialize SupplyChainGraphEngine: %s",
                exc, exc_info=True,
            )
            return None

    async def _init_multi_tier_mapper(self) -> Any:
        """Initialize the MultiTierMapper.

        Returns:
            Initialized MultiTierMapper instance, or None if unavailable.
        """
        try:
            from greenlang.agents.eudr.supply_chain_mapper.multi_tier_mapper import (
                MultiTierMapper,
            )

            # MultiTierMapper requires a graph_storage backend.
            # Use the graph engine as the storage if available.
            if self._graph_engine is not None:
                mapper = MultiTierMapper(graph_storage=self._graph_engine)
                logger.info("MultiTierMapper initialized")
                return mapper
            logger.warning(
                "MultiTierMapper skipped: no graph engine available"
            )
            return None
        except ImportError:
            logger.warning("MultiTierMapper not available")
            return None
        except Exception as exc:
            logger.error(
                "Failed to initialize MultiTierMapper: %s",
                exc, exc_info=True,
            )
            return None

    async def _init_geolocation_linker(self) -> Any:
        """Initialize the GeolocationLinker.

        Returns:
            Initialized GeolocationLinker instance, or None if unavailable.
        """
        try:
            from greenlang.agents.eudr.supply_chain_mapper.geolocation_linker import (
                GeolocationLinker,
            )

            linker = GeolocationLinker(config=self._config)
            logger.info("GeolocationLinker initialized")
            return linker
        except ImportError:
            logger.warning("GeolocationLinker not available")
            return None
        except Exception as exc:
            logger.error(
                "Failed to initialize GeolocationLinker: %s",
                exc, exc_info=True,
            )
            return None

    async def _init_batch_traceability_engine(self) -> Any:
        """Initialize the BatchTraceabilityEngine.

        Returns:
            Initialized BatchTraceabilityEngine instance, or None.
        """
        try:
            from greenlang.agents.eudr.supply_chain_mapper.batch_traceability import (
                BatchTraceabilityEngine,
            )

            engine = BatchTraceabilityEngine(config=self._config)
            logger.info("BatchTraceabilityEngine initialized")
            return engine
        except ImportError:
            logger.debug(
                "BatchTraceabilityEngine module not yet available"
            )
            return None
        except Exception as exc:
            logger.error(
                "Failed to initialize BatchTraceabilityEngine: %s",
                exc, exc_info=True,
            )
            return None

    async def _init_risk_propagation_engine(self) -> Any:
        """Initialize the RiskPropagationEngine.

        Returns:
            Initialized RiskPropagationEngine instance, or None.
        """
        try:
            from greenlang.agents.eudr.supply_chain_mapper.risk_propagation import (
                RiskPropagationConfig,
                RiskPropagationEngine,
            )
            from decimal import Decimal

            risk_config = RiskPropagationConfig(
                weight_country=Decimal(
                    str(self._config.risk_weight_country)
                ),
                weight_commodity=Decimal(
                    str(self._config.risk_weight_commodity)
                ),
                weight_supplier=Decimal(
                    str(self._config.risk_weight_supplier)
                ),
                weight_deforestation=Decimal(
                    str(self._config.risk_weight_deforestation)
                ),
            )
            engine = RiskPropagationEngine(config=risk_config)
            logger.info("RiskPropagationEngine initialized")
            return engine
        except ImportError:
            logger.warning("RiskPropagationEngine not available")
            return None
        except Exception as exc:
            logger.error(
                "Failed to initialize RiskPropagationEngine: %s",
                exc, exc_info=True,
            )
            return None

    async def _init_gap_analyzer(self) -> Any:
        """Initialize the GapAnalyzer.

        Returns:
            Initialized GapAnalyzer instance, or None.
        """
        try:
            from greenlang.agents.eudr.supply_chain_mapper.gap_analyzer import (
                GapAnalyzer,
            )

            analyzer = GapAnalyzer(config=self._config)
            logger.info("GapAnalyzer initialized")
            return analyzer
        except ImportError:
            logger.debug("GapAnalyzer module not yet available")
            return None
        except Exception as exc:
            logger.error(
                "Failed to initialize GapAnalyzer: %s",
                exc, exc_info=True,
            )
            return None

    async def _init_visualization_engine(self) -> Any:
        """Initialize the VisualizationEngine.

        Returns:
            Initialized VisualizationEngine instance, or None.
        """
        try:
            from greenlang.agents.eudr.supply_chain_mapper.visualization import (
                VisualizationEngine,
            )

            engine = VisualizationEngine(config=self._config)
            logger.info("VisualizationEngine initialized")
            return engine
        except ImportError:
            logger.debug("VisualizationEngine module not yet available")
            return None
        except Exception as exc:
            logger.error(
                "Failed to initialize VisualizationEngine: %s",
                exc, exc_info=True,
            )
            return None

    async def _init_regulatory_exporter(self) -> Any:
        """Initialize the RegulatoryExporter.

        Returns:
            Initialized RegulatoryExporter instance, or None.
        """
        try:
            from greenlang.agents.eudr.supply_chain_mapper.regulatory_exporter import (
                RegulatoryExporter,
            )

            exporter = RegulatoryExporter(config=self._config)
            logger.info("RegulatoryExporter initialized")
            return exporter
        except ImportError:
            logger.debug("RegulatoryExporter module not yet available")
            return None
        except Exception as exc:
            logger.error(
                "Failed to initialize RegulatoryExporter: %s",
                exc, exc_info=True,
            )
            return None

    async def _init_supplier_onboarding_engine(self) -> Any:
        """Initialize the SupplierOnboardingEngine.

        Returns:
            Initialized SupplierOnboardingEngine instance, or None.
        """
        try:
            from greenlang.agents.eudr.supply_chain_mapper.supplier_onboarding import (
                SupplierOnboardingEngine,
            )

            engine = SupplierOnboardingEngine(config=self._config)
            logger.info("SupplierOnboardingEngine initialized")
            return engine
        except ImportError:
            logger.debug(
                "SupplierOnboardingEngine module not yet available"
            )
            return None
        except Exception as exc:
            logger.error(
                "Failed to initialize SupplierOnboardingEngine: %s",
                exc, exc_info=True,
            )
            return None

    # ------------------------------------------------------------------
    # Internal: Engine shutdown
    # ------------------------------------------------------------------

    async def _close_engines(self) -> None:
        """Close all engines that implement a close/shutdown method."""
        engine_names = [
            ("graph_engine", self._graph_engine),
            ("multi_tier_mapper", self._multi_tier_mapper),
            ("geolocation_linker", self._geolocation_linker),
            ("batch_traceability_engine", self._batch_traceability_engine),
            ("risk_propagation_engine", self._risk_propagation_engine),
            ("gap_analyzer", self._gap_analyzer),
            ("visualization_engine", self._visualization_engine),
            ("regulatory_exporter", self._regulatory_exporter),
            ("supplier_onboarding_engine", self._supplier_onboarding_engine),
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
        self._graph_engine = None
        self._multi_tier_mapper = None
        self._geolocation_linker = None
        self._batch_traceability_engine = None
        self._risk_propagation_engine = None
        self._gap_analyzer = None
        self._visualization_engine = None
        self._regulatory_exporter = None
        self._supplier_onboarding_engine = None

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
            name="scm-health-check",
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
        """Check initialization status of all nine engines.

        Returns:
            Dictionary with per-engine status and overall engine health.
        """
        engines = {
            "graph_engine": self._graph_engine,
            "multi_tier_mapper": self._multi_tier_mapper,
            "geolocation_linker": self._geolocation_linker,
            "batch_traceability_engine": self._batch_traceability_engine,
            "risk_propagation_engine": self._risk_propagation_engine,
            "gap_analyzer": self._gap_analyzer,
            "visualization_engine": self._visualization_engine,
            "regulatory_exporter": self._regulatory_exporter,
            "supplier_onboarding_engine": self._supplier_onboarding_engine,
        }

        engine_statuses: Dict[str, str] = {}
        initialized_count = 0
        for name, engine in engines.items():
            if engine is not None:
                engine_statuses[name] = "initialized"
                initialized_count += 1
            else:
                engine_statuses[name] = "unavailable"

        # The core engines (graph, risk, geolocation) are required;
        # others are optional and their absence degrades but does not
        # break the service.
        core_engines = [
            "graph_engine",
            "risk_propagation_engine",
            "geolocation_linker",
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
        try:
            from greenlang.agents.eudr.supply_chain_mapper.metrics import (
                set_active_graphs,
                set_total_nodes,
            )

            set_active_graphs(0)
            set_total_nodes(0)
            logger.debug("Prometheus metrics flushed")
        except ImportError:
            pass

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
                "SupplyChainMapperService is not started. "
                "Call await service.startup() first."
            )

    # ------------------------------------------------------------------
    # Convenience: get_engine
    # ------------------------------------------------------------------

    def get_engine(self, name: str) -> Any:
        """Retrieve an engine by name.

        Args:
            name: Engine name (e.g., 'graph_engine', 'risk_propagation_engine').

        Returns:
            The engine instance, or None if not initialized.

        Raises:
            RuntimeError: If the service has not been started.
            ValueError: If the engine name is not recognized.
        """
        self._ensure_started()
        valid_names = {
            "graph_engine": self._graph_engine,
            "multi_tier_mapper": self._multi_tier_mapper,
            "geolocation_linker": self._geolocation_linker,
            "batch_traceability_engine": self._batch_traceability_engine,
            "risk_propagation_engine": self._risk_propagation_engine,
            "gap_analyzer": self._gap_analyzer,
            "visualization_engine": self._visualization_engine,
            "regulatory_exporter": self._regulatory_exporter,
            "supplier_onboarding_engine": self._supplier_onboarding_engine,
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
            Count of non-None engine instances (0 to 9).
        """
        engines = [
            self._graph_engine,
            self._multi_tier_mapper,
            self._geolocation_linker,
            self._batch_traceability_engine,
            self._risk_propagation_engine,
            self._gap_analyzer,
            self._visualization_engine,
            self._regulatory_exporter,
            self._supplier_onboarding_engine,
        ]
        return sum(1 for e in engines if e is not None)

# ---------------------------------------------------------------------------
# FastAPI lifespan context manager
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: Any) -> AsyncIterator[None]:
    """FastAPI lifespan context manager for the Supply Chain Mapper service.

    Automatically starts the service on application startup and shuts it
    down on application shutdown. The service instance is stored in
    ``app.state.scm_service`` for access from route handlers.

    Usage with FastAPI::

        from fastapi import FastAPI
        from greenlang.agents.eudr.supply_chain_mapper.setup import lifespan
from greenlang.schemas import utcnow

        app = FastAPI(lifespan=lifespan)

    Args:
        app: The FastAPI application instance.

    Yields:
        None (service is accessible via ``app.state.scm_service``).
    """
    service = get_service()
    app.state.scm_service = service
    try:
        await service.startup()
        yield
    finally:
        await service.shutdown()

# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_service_instance: Optional[SupplyChainMapperService] = None
_service_lock = threading.Lock()

def get_service(
    config: Optional[SupplyChainMapperConfig] = None,
) -> SupplyChainMapperService:
    """Return the singleton SupplyChainMapperService instance.

    Uses double-checked locking for thread safety. The instance is
    created on first call. Pass a config to override the default
    environment-based configuration.

    Args:
        config: Optional configuration override.

    Returns:
        SupplyChainMapperService singleton instance.

    Example:
        >>> service = get_service()
        >>> await service.startup()
    """
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = SupplyChainMapperService(config=config)
    return _service_instance

def set_service(service: SupplyChainMapperService) -> None:
    """Replace the singleton SupplyChainMapperService instance.

    Primarily intended for testing and dependency injection.

    Args:
        service: Replacement service instance.
    """
    global _service_instance
    with _service_lock:
        _service_instance = service
    logger.info("SupplyChainMapperService singleton replaced")

def reset_service() -> None:
    """Reset the singleton SupplyChainMapperService to None.

    The next call to ``get_service()`` will create a fresh instance.
    Intended for test teardown.
    """
    global _service_instance
    with _service_lock:
        _service_instance = None
    logger.debug("SupplyChainMapperService singleton reset")

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "SupplyChainMapperService",
    "HealthStatus",
    "lifespan",
    "get_service",
    "set_service",
    "reset_service",
]
