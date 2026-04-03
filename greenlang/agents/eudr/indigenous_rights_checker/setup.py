# -*- coding: utf-8 -*-
"""
IndigenousRightsCheckerService - Facade for AGENT-EUDR-021

Unified setup facade orchestrating all 7 engines of the Indigenous
Rights Checker Agent. Provides a single entry point for territory
database management, FPIC verification, land rights overlap detection,
community consultation tracking, rights violation monitoring,
indigenous community registry, and compliance reporting.

Engines (7):
    1. TerritoryDatabaseEngine      - Indigenous territory data management (Feature 1)
    2. FPICVerificationEngine        - FPIC documentation verification (Feature 2)
    3. LandRightsOverlapEngine       - PostGIS overlap detection (Feature 3)
    4. CommunityConsultationEngine   - Community engagement tracking (Feature 4)
    5. RightsViolationEngine         - Violation monitoring & scoring (Feature 5)
    6. IndigenousRegistryEngine      - Community database (Feature 6)
    7. ComplianceReportingEngine     - Report generation (Feature 8)

Reference Data (3):
    - ilo_169_countries: 24 ILO Convention 169 ratifying countries
    - indigenous_territory_sources: 6 authoritative data sources
    - fpic_legal_frameworks: 8 country FPIC legal frameworks

Singleton Pattern:
    Thread-safe singleton with double-checked locking via ``get_service()``.

FastAPI Integration:
    Use the ``lifespan`` async context manager with
    ``FastAPI(lifespan=lifespan)`` for automatic startup/shutdown.

Example:
    >>> from greenlang.agents.eudr.indigenous_rights_checker.setup import (
    ...     IndigenousRightsCheckerService,
    ...     get_service,
    ... )
    >>> service = get_service()
    >>> await service.startup()
    >>> health = await service.health_check()
    >>> assert health["status"] == "healthy"
    >>>
    >>> await service.shutdown()

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-021 Indigenous Rights Checker (GL-EUDR-IRC-021)
Regulation: EU 2023/1115 (EUDR) Articles 2, 8, 10, 11, 29, 31
Status: Production Ready
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional
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

from greenlang.agents.eudr.indigenous_rights_checker.config import (
    IndigenousRightsCheckerConfig,
    get_config,
)
from greenlang.agents.eudr.indigenous_rights_checker.provenance import (
    ProvenanceTracker,
    get_tracker,
)
from greenlang.agents.eudr.indigenous_rights_checker.metrics import (
    PROMETHEUS_AVAILABLE,
    record_api_error,
    set_active_territories,
    set_active_overlaps,
    set_active_workflows,
)

# ---------------------------------------------------------------------------
# Engine imports (graceful fallback for lazy loading)
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.indigenous_rights_checker.territory_database_engine import (
        TerritoryDatabaseEngine,
    )
    _TERRITORY_ENGINE_AVAILABLE = True
except ImportError:
    TerritoryDatabaseEngine = None  # type: ignore[assignment,misc]
    _TERRITORY_ENGINE_AVAILABLE = False

try:
    from greenlang.agents.eudr.indigenous_rights_checker.fpic_verification_engine import (
        FPICVerificationEngine,
    )
    _FPIC_ENGINE_AVAILABLE = True
except ImportError:
    FPICVerificationEngine = None  # type: ignore[assignment,misc]
    _FPIC_ENGINE_AVAILABLE = False

try:
    from greenlang.agents.eudr.indigenous_rights_checker.land_rights_overlap_engine import (
        LandRightsOverlapEngine,
    )
    _OVERLAP_ENGINE_AVAILABLE = True
except ImportError:
    LandRightsOverlapEngine = None  # type: ignore[assignment,misc]
    _OVERLAP_ENGINE_AVAILABLE = False

try:
    from greenlang.agents.eudr.indigenous_rights_checker.community_consultation_engine import (
        CommunityConsultationEngine,
    )
    _CONSULTATION_ENGINE_AVAILABLE = True
except ImportError:
    CommunityConsultationEngine = None  # type: ignore[assignment,misc]
    _CONSULTATION_ENGINE_AVAILABLE = False

try:
    from greenlang.agents.eudr.indigenous_rights_checker.rights_violation_engine import (
        RightsViolationEngine,
    )
    _VIOLATION_ENGINE_AVAILABLE = True
except ImportError:
    RightsViolationEngine = None  # type: ignore[assignment,misc]
    _VIOLATION_ENGINE_AVAILABLE = False

try:
    from greenlang.agents.eudr.indigenous_rights_checker.indigenous_registry_engine import (
        IndigenousRegistryEngine,
    )
    _REGISTRY_ENGINE_AVAILABLE = True
except ImportError:
    IndigenousRegistryEngine = None  # type: ignore[assignment,misc]
    _REGISTRY_ENGINE_AVAILABLE = False

try:
    from greenlang.agents.eudr.indigenous_rights_checker.compliance_reporting_engine import (
        ComplianceReportingEngine,
    )
    _REPORTING_ENGINE_AVAILABLE = True
except ImportError:
    ComplianceReportingEngine = None  # type: ignore[assignment,misc]
    _REPORTING_ENGINE_AVAILABLE = False

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"
_AGENT_ID = "GL-EUDR-IRC-021"
_ENGINE_COUNT = 7
_ENGINE_NAMES: List[str] = [
    "TerritoryDatabaseEngine",
    "FPICVerificationEngine",
    "LandRightsOverlapEngine",
    "CommunityConsultationEngine",
    "RightsViolationEngine",
    "IndigenousRegistryEngine",
    "ComplianceReportingEngine",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# =============================================================================
# FACADE: IndigenousRightsCheckerService
# =============================================================================

class IndigenousRightsCheckerService:
    """IndigenousRightsCheckerService orchestrates all 7 engines of AGENT-EUDR-021.

    This facade provides a single, thread-safe entry point for all
    indigenous rights checking operations per EUDR Articles 2, 8, 10,
    11, 29, and 31.

    Architecture:
        - Lazy initialization of all 7 engines (on first use)
        - Thread-safe singleton pattern with double-checked locking
        - PostgreSQL + PostGIS connection pooling via psycopg_pool
        - Redis caching for frequently accessed reference data
        - OpenTelemetry distributed tracing integration
        - Prometheus metrics for all operations
        - SHA-256 provenance hashing for audit trails

    Engines:
        1. TerritoryDatabaseEngine: Indigenous territory CRUD with PostGIS,
           multi-source data ingestion (LandMark, RAISG, FUNAI, BPN/AMAN,
           ACHPR, national registries), versioning, and freshness tracking
        2. FPICVerificationEngine: 10-element weighted FPIC scoring with
           Decimal arithmetic, country-specific rules, temporal compliance,
           coercion detection, and validity period management
        3. LandRightsOverlapEngine: PostGIS overlap detection using
           ST_Contains, ST_DWithin, ST_Distance, batch screening of up
           to 10,000 plots, 4-tier overlap classification (direct,
           partial, adjacent, proximate)
        4. CommunityConsultationEngine: 7-stage consultation lifecycle
           tracking, grievance management with SLA deadlines, benefit-
           sharing agreement management
        5. RightsViolationEngine: Violation monitoring from 10+ sources,
           5-factor severity scoring with Decimal arithmetic, 7-day
           deduplication, supply chain correlation via PostGIS proximity
        6. IndigenousRegistryEngine: Community database with auto-populated
           ILO 169 coverage and FPIC requirements, legal protections,
           commodity relevance, and privacy controls
        7. ComplianceReportingEngine: 8 report types in 5 formats and
           5 languages, DDS sections, certification scheme reports,
           BI exports with full provenance

    Attributes:
        config: Current configuration instance.
        provenance_tracker: SHA-256 provenance tracking.

    Example:
        >>> service = get_service()
        >>> await service.startup()
        >>> health = await service.health_check()
        >>> assert health["status"] == "healthy"
        >>> await service.shutdown()
    """

    def __init__(
        self,
        config: Optional[IndigenousRightsCheckerConfig] = None,
        *,
        db_pool: Optional[Any] = None,
        redis_client: Optional[Any] = None,
    ) -> None:
        """Initialize IndigenousRightsCheckerService.

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
            self._tracer = otel_trace.get_tracer(
                __name__, version=_MODULE_VERSION
            )
        else:
            self._tracer = None

        # Engine instances (lazy initialized)
        self._territory_engine: Optional[Any] = None
        self._fpic_engine: Optional[Any] = None
        self._overlap_engine: Optional[Any] = None
        self._consultation_engine: Optional[Any] = None
        self._violation_engine: Optional[Any] = None
        self._registry_engine: Optional[Any] = None
        self._reporting_engine: Optional[Any] = None

        # Lifecycle state
        self._started: bool = False
        self._startup_time: Optional[datetime] = None
        self._startup_lock = asyncio.Lock()
        self._shutdown_lock = asyncio.Lock()

        # Statistics tracking
        self._stats: Dict[str, int] = {
            "total_territory_queries": 0,
            "total_fpic_assessments": 0,
            "total_overlap_detections": 0,
            "total_consultations": 0,
            "total_violations_ingested": 0,
            "total_community_registrations": 0,
            "total_reports_generated": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
        }

        logger.info(
            f"IndigenousRightsCheckerService initialized "
            f"(version={_MODULE_VERSION}, agent_id={_AGENT_ID})"
        )

    # -----------------------------------------------------------------------
    # Properties for engine access
    # -----------------------------------------------------------------------

    @property
    def territory_engine(self) -> Any:
        """Access TerritoryDatabaseEngine (lazy initialized).

        Returns:
            TerritoryDatabaseEngine instance.

        Raises:
            RuntimeError: If engine is not available.
        """
        if self._territory_engine is None:
            raise RuntimeError(
                "TerritoryDatabaseEngine not initialized. "
                "Call startup() first."
            )
        return self._territory_engine

    @property
    def fpic_engine(self) -> Any:
        """Access FPICVerificationEngine (lazy initialized).

        Returns:
            FPICVerificationEngine instance.

        Raises:
            RuntimeError: If engine is not available.
        """
        if self._fpic_engine is None:
            raise RuntimeError(
                "FPICVerificationEngine not initialized. "
                "Call startup() first."
            )
        return self._fpic_engine

    @property
    def overlap_engine(self) -> Any:
        """Access LandRightsOverlapEngine (lazy initialized).

        Returns:
            LandRightsOverlapEngine instance.

        Raises:
            RuntimeError: If engine is not available.
        """
        if self._overlap_engine is None:
            raise RuntimeError(
                "LandRightsOverlapEngine not initialized. "
                "Call startup() first."
            )
        return self._overlap_engine

    @property
    def consultation_engine(self) -> Any:
        """Access CommunityConsultationEngine (lazy initialized).

        Returns:
            CommunityConsultationEngine instance.

        Raises:
            RuntimeError: If engine is not available.
        """
        if self._consultation_engine is None:
            raise RuntimeError(
                "CommunityConsultationEngine not initialized. "
                "Call startup() first."
            )
        return self._consultation_engine

    @property
    def violation_engine(self) -> Any:
        """Access RightsViolationEngine (lazy initialized).

        Returns:
            RightsViolationEngine instance.

        Raises:
            RuntimeError: If engine is not available.
        """
        if self._violation_engine is None:
            raise RuntimeError(
                "RightsViolationEngine not initialized. "
                "Call startup() first."
            )
        return self._violation_engine

    @property
    def registry_engine(self) -> Any:
        """Access IndigenousRegistryEngine (lazy initialized).

        Returns:
            IndigenousRegistryEngine instance.

        Raises:
            RuntimeError: If engine is not available.
        """
        if self._registry_engine is None:
            raise RuntimeError(
                "IndigenousRegistryEngine not initialized. "
                "Call startup() first."
            )
        return self._registry_engine

    @property
    def reporting_engine(self) -> Any:
        """Access ComplianceReportingEngine (lazy initialized).

        Returns:
            ComplianceReportingEngine instance.

        Raises:
            RuntimeError: If engine is not available.
        """
        if self._reporting_engine is None:
            raise RuntimeError(
                "ComplianceReportingEngine not initialized. "
                "Call startup() first."
            )
        return self._reporting_engine

    # -----------------------------------------------------------------------
    # Lifecycle management
    # -----------------------------------------------------------------------

    async def startup(self) -> None:
        """Initialize all resources (database, Redis, engines).

        This method is idempotent and thread-safe. Multiple calls are safe.

        Raises:
            RuntimeError: If startup fails critically.
        """
        async with self._startup_lock:
            if self._started:
                logger.debug(
                    "IndigenousRightsCheckerService already started, "
                    "skipping startup"
                )
                return

            logger.info("Starting IndigenousRightsCheckerService...")
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
                if (
                    self._redis_client is None
                    and REDIS_AVAILABLE
                    and aioredis
                ):
                    await self._init_redis()
                else:
                    logger.info(
                        "Redis disabled or redis library not available"
                    )

                # 3. Initialize all engines
                await self._init_engines()

                # 4. Mark as started
                self._started = True
                self._startup_time = utcnow()
                duration_ms = (time.monotonic() - start_time) * 1000

                # 5. Log provenance
                self._provenance_tracker.record(
                    "config_change", "create", _AGENT_ID,
                    metadata={
                        "event": "startup",
                        "duration_ms": duration_ms,
                    },
                )

                logger.info(
                    f"IndigenousRightsCheckerService started successfully "
                    f"(duration={duration_ms:.2f}ms)"
                )

            except Exception as e:
                logger.error("Startup failed: %s", e, exc_info=True)
                record_api_error("startup", "500")
                raise RuntimeError(f"Service startup failed: {e}") from e

    async def shutdown(self) -> None:
        """Gracefully shutdown all resources.

        Closes database pool, Redis client, and cleans up engine resources.
        This method is idempotent and safe to call multiple times.
        """
        async with self._shutdown_lock:
            if not self._started:
                logger.debug(
                    "IndigenousRightsCheckerService not started, "
                    "skipping shutdown"
                )
                return

            logger.info("Shutting down IndigenousRightsCheckerService...")

            try:
                # 1. Shutdown engines
                await self._shutdown_engines()

                # 2. Close Redis client
                if self._redis_client is not None:
                    try:
                        await self._redis_client.close()
                        logger.debug("Redis client closed")
                    except Exception as e:
                        logger.warning("Error closing Redis client: %s", e)

                # 3. Close database pool
                if self._db_pool is not None:
                    try:
                        await self._db_pool.close()
                        logger.debug("PostgreSQL pool closed")
                    except Exception as e:
                        logger.warning(
                            f"Error closing PostgreSQL pool: {e}"
                        )

                # 4. Mark as shutdown
                self._started = False

                self._provenance_tracker.record(
                    "config_change", "update", _AGENT_ID,
                    metadata={"event": "shutdown"},
                )

                logger.info(
                    "IndigenousRightsCheckerService shutdown complete"
                )

            except Exception as e:
                logger.error("Shutdown error: %s", e, exc_info=True)

    async def health_check(
        self, include_details: bool = False
    ) -> Dict[str, Any]:
        """Perform a health check on all service components.

        Args:
            include_details: Include detailed engine status.

        Returns:
            Health status dictionary.
        """
        health: Dict[str, Any] = {
            "status": "healthy" if self._started else "unavailable",
            "agent_id": _AGENT_ID,
            "version": _MODULE_VERSION,
            "started": self._started,
            "startup_time": (
                self._startup_time.isoformat()
                if self._startup_time
                else None
            ),
            "database_connected": self._db_pool is not None,
            "redis_connected": self._redis_client is not None,
        }

        if include_details and self._started:
            health["engines"] = {
                "territory_database": self._territory_engine is not None,
                "fpic_verification": self._fpic_engine is not None,
                "land_rights_overlap": self._overlap_engine is not None,
                "community_consultation": (
                    self._consultation_engine is not None
                ),
                "rights_violation": self._violation_engine is not None,
                "indigenous_registry": self._registry_engine is not None,
                "compliance_reporting": self._reporting_engine is not None,
            }
            health["statistics"] = dict(self._stats)
            health["provenance_chain_length"] = len(
                self._provenance_tracker.get_chain()
            )
            health["provenance_chain_valid"] = (
                self._provenance_tracker.verify_chain()
            )

            # Get live counts if database connected
            if self._db_pool is not None:
                try:
                    counts = await self._get_live_counts()
                    health["territory_count"] = counts.get("territories", 0)
                    health["community_count"] = counts.get("communities", 0)
                    health["active_workflows"] = counts.get("workflows", 0)
                    health["active_violations"] = counts.get(
                        "violations", 0
                    )
                except Exception as e:
                    logger.warning("Error fetching live counts: %s", e)

        return health

    # -----------------------------------------------------------------------
    # Resource initialization
    # -----------------------------------------------------------------------

    async def _init_db_pool(self) -> None:
        """Initialize PostgreSQL connection pool with PostGIS support."""
        try:
            self._db_pool = AsyncConnectionPool(
                conninfo=self._config.database_url,
                min_size=2,
                max_size=self._config.pool_size,
                max_idle=self._config.pool_recycle_s,
                timeout=float(self._config.pool_timeout_s),
                open=False,
            )
            await self._db_pool.open()
            logger.info(
                f"PostgreSQL pool initialized "
                f"(size={self._config.pool_size})"
            )
        except Exception as e:
            logger.error("Failed to initialize PostgreSQL pool: %s", e)
            self._db_pool = None

    async def _init_redis(self) -> None:
        """Initialize Redis async client."""
        try:
            self._redis_client = aioredis.from_url(
                self._config.redis_url,
                decode_responses=True,
            )
            await self._redis_client.ping()
            logger.info("Redis client initialized and connected")
        except Exception as e:
            logger.error("Failed to initialize Redis client: %s", e)
            self._redis_client = None

    async def _init_engines(self) -> None:
        """Initialize all 7 engines."""
        engine_inits = [
            ("TerritoryDatabaseEngine", self._init_territory_engine),
            ("FPICVerificationEngine", self._init_fpic_engine),
            ("LandRightsOverlapEngine", self._init_overlap_engine),
            ("CommunityConsultationEngine", self._init_consultation_engine),
            ("RightsViolationEngine", self._init_violation_engine),
            ("IndigenousRegistryEngine", self._init_registry_engine),
            ("ComplianceReportingEngine", self._init_reporting_engine),
        ]

        for engine_name, init_method in engine_inits:
            try:
                await init_method()
                logger.info("Engine %s initialized successfully", engine_name)
            except Exception as e:
                logger.error(
                    f"Failed to initialize {engine_name}: {e}",
                    exc_info=True,
                )

    async def _init_territory_engine(self) -> None:
        """Initialize TerritoryDatabaseEngine."""
        if not _TERRITORY_ENGINE_AVAILABLE or TerritoryDatabaseEngine is None:
            logger.warning("TerritoryDatabaseEngine not available")
            return
        self._territory_engine = TerritoryDatabaseEngine(
            self._config, self._provenance_tracker
        )
        if self._db_pool:
            await self._territory_engine.startup(self._db_pool)

    async def _init_fpic_engine(self) -> None:
        """Initialize FPICVerificationEngine."""
        if not _FPIC_ENGINE_AVAILABLE or FPICVerificationEngine is None:
            logger.warning("FPICVerificationEngine not available")
            return
        self._fpic_engine = FPICVerificationEngine(
            self._config, self._provenance_tracker
        )
        if self._db_pool:
            await self._fpic_engine.startup(self._db_pool)

    async def _init_overlap_engine(self) -> None:
        """Initialize LandRightsOverlapEngine."""
        if not _OVERLAP_ENGINE_AVAILABLE or LandRightsOverlapEngine is None:
            logger.warning("LandRightsOverlapEngine not available")
            return
        self._overlap_engine = LandRightsOverlapEngine(
            self._config, self._provenance_tracker
        )
        if self._db_pool:
            await self._overlap_engine.startup(self._db_pool)

    async def _init_consultation_engine(self) -> None:
        """Initialize CommunityConsultationEngine."""
        if (
            not _CONSULTATION_ENGINE_AVAILABLE
            or CommunityConsultationEngine is None
        ):
            logger.warning("CommunityConsultationEngine not available")
            return
        self._consultation_engine = CommunityConsultationEngine(
            self._config, self._provenance_tracker
        )
        if self._db_pool:
            await self._consultation_engine.startup(self._db_pool)

    async def _init_violation_engine(self) -> None:
        """Initialize RightsViolationEngine."""
        if not _VIOLATION_ENGINE_AVAILABLE or RightsViolationEngine is None:
            logger.warning("RightsViolationEngine not available")
            return
        self._violation_engine = RightsViolationEngine(
            self._config, self._provenance_tracker
        )
        if self._db_pool:
            await self._violation_engine.startup(self._db_pool)

    async def _init_registry_engine(self) -> None:
        """Initialize IndigenousRegistryEngine."""
        if not _REGISTRY_ENGINE_AVAILABLE or IndigenousRegistryEngine is None:
            logger.warning("IndigenousRegistryEngine not available")
            return
        self._registry_engine = IndigenousRegistryEngine(
            self._config, self._provenance_tracker
        )
        if self._db_pool:
            await self._registry_engine.startup(self._db_pool)

    async def _init_reporting_engine(self) -> None:
        """Initialize ComplianceReportingEngine."""
        if (
            not _REPORTING_ENGINE_AVAILABLE
            or ComplianceReportingEngine is None
        ):
            logger.warning("ComplianceReportingEngine not available")
            return
        self._reporting_engine = ComplianceReportingEngine(
            self._config, self._provenance_tracker
        )
        if self._db_pool:
            await self._reporting_engine.startup(self._db_pool)

    async def _shutdown_engines(self) -> None:
        """Shutdown all initialized engines."""
        engines = [
            ("TerritoryDatabaseEngine", self._territory_engine),
            ("FPICVerificationEngine", self._fpic_engine),
            ("LandRightsOverlapEngine", self._overlap_engine),
            ("CommunityConsultationEngine", self._consultation_engine),
            ("RightsViolationEngine", self._violation_engine),
            ("IndigenousRegistryEngine", self._registry_engine),
            ("ComplianceReportingEngine", self._reporting_engine),
        ]

        for engine_name, engine in engines:
            if engine is not None:
                try:
                    await engine.shutdown()
                    logger.debug("Engine %s shutdown", engine_name)
                except Exception as e:
                    logger.warning(
                        f"Error shutting down {engine_name}: {e}"
                    )

        # Clear references
        self._territory_engine = None
        self._fpic_engine = None
        self._overlap_engine = None
        self._consultation_engine = None
        self._violation_engine = None
        self._registry_engine = None
        self._reporting_engine = None

    # -----------------------------------------------------------------------
    # Live data counts
    # -----------------------------------------------------------------------

    async def _get_live_counts(self) -> Dict[str, int]:
        """Get live entity counts from database.

        Returns:
            Dictionary with entity counts.
        """
        counts: Dict[str, int] = {}

        if self._db_pool is None:
            return counts

        queries = {
            "territories": (
                "SELECT COUNT(*) FROM "
                "eudr_indigenous_rights_checker.gl_eudr_irc_territories"
            ),
            "communities": (
                "SELECT COUNT(*) FROM "
                "eudr_indigenous_rights_checker.gl_eudr_irc_communities"
            ),
            "workflows": (
                "SELECT COUNT(*) FROM "
                "eudr_indigenous_rights_checker.gl_eudr_irc_workflows "
                "WHERE current_stage NOT IN "
                "('consent_withdrawn', 'consent_denied')"
            ),
            "violations": (
                "SELECT COUNT(*) FROM "
                "eudr_indigenous_rights_checker."
                "gl_eudr_irc_violation_alerts "
                "WHERE status = 'active'"
            ),
        }

        async with self._db_pool.connection() as conn:
            async with conn.cursor() as cur:
                for key, query in queries.items():
                    try:
                        await cur.execute(query)
                        row = await cur.fetchone()
                        counts[key] = row[0] if row else 0
                    except Exception as e:
                        logger.warning(
                            f"Error counting {key}: {e}"
                        )
                        counts[key] = 0

        # Update gauges
        set_active_territories(counts.get("territories", 0))
        set_active_overlaps(0)
        set_active_workflows(counts.get("workflows", 0))

        return counts

# ---------------------------------------------------------------------------
# Thread-safe singleton pattern (double-checked locking)
# ---------------------------------------------------------------------------

_service_lock = threading.Lock()
_global_service: Optional[IndigenousRightsCheckerService] = None

def get_service(
    config: Optional[IndigenousRightsCheckerConfig] = None,
) -> IndigenousRightsCheckerService:
    """Get the global IndigenousRightsCheckerService singleton instance.

    Thread-safe lazy initialization. Subsequent calls return the same
    instance. Pass a config to override the default configuration on
    first creation.

    Args:
        config: Optional configuration override for first creation.

    Returns:
        IndigenousRightsCheckerService singleton instance.

    Example:
        >>> service = get_service()
        >>> service2 = get_service()
        >>> assert service is service2
    """
    global _global_service
    if _global_service is None:
        with _service_lock:
            if _global_service is None:
                _global_service = IndigenousRightsCheckerService(
                    config=config
                )
    return _global_service

def reset_service() -> None:
    """Reset the global service singleton (for testing only).

    WARNING: Only use in test environments.
    """
    global _global_service
    with _service_lock:
        _global_service = None
        logger.warning("Service singleton reset (testing only)")

# ---------------------------------------------------------------------------
# FastAPI lifespan context manager
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: Any) -> AsyncIterator[None]:
    """FastAPI lifespan context manager for startup/shutdown.

    Usage:
        >>> from fastapi import FastAPI
        >>> from greenlang.agents.eudr.indigenous_rights_checker.setup import lifespan
        >>> app = FastAPI(lifespan=lifespan)

    Args:
        app: FastAPI application instance.

    Yields:
        None (service is available via get_service() during lifespan).
    """
    service = get_service()
    await service.startup()
    logger.info("IndigenousRightsCheckerService lifespan started")
    try:
        yield
    finally:
        await service.shutdown()
        reset_service()
        logger.info("IndigenousRightsCheckerService lifespan ended")
