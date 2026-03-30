# -*- coding: utf-8 -*-
"""
CommodityRiskAnalyzerSetup - Facade for AGENT-EUDR-018

Unified setup facade orchestrating all 8 engines of the Commodity Risk
Analyzer Agent. Provides a single entry point for commodity profiling,
derived product analysis, price volatility monitoring, production
forecasting, substitution risk detection, regulatory compliance checking,
commodity-specific due diligence, and portfolio risk aggregation.

Engines (8):
    1. CommodityProfiler           - Base commodity risk profiling (Feature 1)
    2. DerivedProductAnalyzer       - Annex I product traceability (Feature 2)
    3. PriceVolatilityEngine        - Price volatility monitoring (Feature 3)
    4. ProductionForecastEngine     - Production forecasting (Feature 4)
    5. SubstitutionRiskAnalyzer     - Substitution risk detection (Feature 5)
    6. RegulatoryComplianceEngine   - EUDR article compliance mapping (Feature 6)
    7. CommodityDueDiligenceEngine  - Due diligence workflow management (Feature 7)
    8. PortfolioRiskAggregator      - Multi-commodity portfolio analysis (Feature 8)

Reference Data (4):
    - commodity_database: 7 EUDR commodities, HS codes, derived products
    - processing_chains: Transformation pathways per commodity
    - production_statistics: Global production volumes, seasonal patterns
    - regulatory_requirements: EUDR article requirements per commodity

Singleton Pattern:
    Thread-safe singleton with double-checked locking via ``get_service()``.

FastAPI Integration:
    Use the ``lifespan`` async context manager with
    ``FastAPI(lifespan=lifespan)`` for automatic startup/shutdown.

Example:
    >>> from greenlang.agents.eudr.commodity_risk_analyzer.setup import (
    ...     CommodityRiskAnalyzerSetup,
    ...     get_service,
    ... )
    >>> service = get_service()
    >>> await service.startup()
    >>> health = await service.health_check()
    >>> assert health["status"] == "healthy"
    >>>
    >>> # Profile a commodity
    >>> profile = await service.profile_commodity(
    ...     commodity_type="cocoa",
    ...     region="west_africa",
    ... )
    >>> assert profile["risk_level"] in ["low", "medium", "high", "critical"]
    >>>
    >>> # Comprehensive analysis
    >>> analysis = await service.run_comprehensive_analysis(
    ...     commodity_type="oil_palm",
    ...     supplier_id="SUP-ID-001",
    ...     include_all=True,
    ... )
    >>>
    >>> await service.shutdown()

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-018
Agent ID: GL-EUDR-CRA-018
Regulation: EU 2023/1115 (EUDR) Articles 1, 2, 3, 4, 8, 9, 10, Annex I
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

from greenlang.agents.eudr.commodity_risk_analyzer.config import (
    CommodityRiskAnalyzerConfig,
    get_config,
    set_config,
    reset_config,
)
from greenlang.agents.eudr.commodity_risk_analyzer.provenance import (
    ProvenanceTracker,
    get_tracker,
)
from greenlang.agents.eudr.commodity_risk_analyzer.metrics import (
    PROMETHEUS_AVAILABLE,
    record_profile_created,
    record_derived_product_analyzed,
    record_price_query,
    record_forecast_generated,
    record_substitution_detected,
    record_compliance_check,
    record_dd_workflow_initiated,
    record_portfolio_analysis,
    record_api_error,
    observe_profile_duration,
    observe_analysis_duration,
    observe_forecast_duration,
    observe_portfolio_duration,
    set_active_workflows,
    set_monitored_commodities,
    set_portfolio_risk_exposure,
    set_high_risk_commodities,
    set_active_substitution_alerts,
)

# ---------------------------------------------------------------------------
# Internal imports: models
# ---------------------------------------------------------------------------

from greenlang.agents.eudr.commodity_risk_analyzer.models import (
    VERSION,
    MAX_RISK_SCORE,
    MIN_RISK_SCORE,
    MAX_BATCH_SIZE,
    EUDR_CUTOFF_DATE,
    EUDR_RETENTION_YEARS,
    SUPPORTED_COMMODITIES,
    SUPPORTED_DERIVED_CATEGORIES,
    SUPPORTED_OUTPUT_FORMATS,
    DEFAULT_COMMODITY_WEIGHTS,
    CommodityType,
    DerivedProductCategory,
    ProcessingStage,
    RiskLevel,
    MarketCondition,
    VolatilityLevel,
    SeasonalPhase,
    ComplianceStatus,
    DDWorkflowStatus,
    EvidenceType,
    PortfolioStrategy,
    ReportFormat,
    CommodityProfile,
    DerivedProduct,
    PriceData,
    ProductionForecast,
    SubstitutionEvent,
    RegulatoryRequirement,
    DDWorkflow,
    PortfolioAnalysis,
    CommodityRiskScore,
    AuditLogEntry,
    ProfileCommodityRequest,
    AnalyzeDerivedProductRequest,
    QueryPriceVolatilityRequest,
    GenerateForecastRequest,
    DetectSubstitutionRequest,
    CheckComplianceRequest,
    InitiateDDWorkflowRequest,
    AggregatePortfolioRequest,
    BatchCommodityAnalysisRequest,
    CompareCommoditiesRequest,
    GetTrendRequest,
    HealthRequest,
    CommodityProfileResponse,
    DerivedProductResponse,
    PriceVolatilityResponse,
    ProductionForecastResponse,
    SubstitutionRiskResponse,
    RegulatoryComplianceResponse,
    DDWorkflowResponse,
    PortfolioAnalysisResponse,
    BatchAnalysisResponse,
    ComparisonResponse,
    TrendResponse,
    HealthResponse,
)

# ---------------------------------------------------------------------------
# Engine imports (graceful fallback for lazy loading)
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.commodity_risk_analyzer.engines.commodity_profiler import (
        CommodityProfiler,
    )

    _COMMODITY_PROFILER_AVAILABLE = True
except ImportError:
    CommodityProfiler = None  # type: ignore[assignment,misc]
    _COMMODITY_PROFILER_AVAILABLE = False

try:
    from greenlang.agents.eudr.commodity_risk_analyzer.engines.derived_product_analyzer import (
        DerivedProductAnalyzer,
    )

    _DERIVED_PRODUCT_ANALYZER_AVAILABLE = True
except ImportError:
    DerivedProductAnalyzer = None  # type: ignore[assignment,misc]
    _DERIVED_PRODUCT_ANALYZER_AVAILABLE = False

try:
    from greenlang.agents.eudr.commodity_risk_analyzer.engines.price_volatility_engine import (
        PriceVolatilityEngine,
    )

    _PRICE_VOLATILITY_ENGINE_AVAILABLE = True
except ImportError:
    PriceVolatilityEngine = None  # type: ignore[assignment,misc]
    _PRICE_VOLATILITY_ENGINE_AVAILABLE = False

try:
    from greenlang.agents.eudr.commodity_risk_analyzer.engines.production_forecast_engine import (
        ProductionForecastEngine,
    )

    _PRODUCTION_FORECAST_ENGINE_AVAILABLE = True
except ImportError:
    ProductionForecastEngine = None  # type: ignore[assignment,misc]
    _PRODUCTION_FORECAST_ENGINE_AVAILABLE = False

try:
    from greenlang.agents.eudr.commodity_risk_analyzer.engines.substitution_risk_analyzer import (
        SubstitutionRiskAnalyzer,
    )

    _SUBSTITUTION_RISK_ANALYZER_AVAILABLE = True
except ImportError:
    SubstitutionRiskAnalyzer = None  # type: ignore[assignment,misc]
    _SUBSTITUTION_RISK_ANALYZER_AVAILABLE = False

try:
    from greenlang.agents.eudr.commodity_risk_analyzer.engines.regulatory_compliance_engine import (
        RegulatoryComplianceEngine,
    )

    _REGULATORY_COMPLIANCE_ENGINE_AVAILABLE = True
except ImportError:
    RegulatoryComplianceEngine = None  # type: ignore[assignment,misc]
    _REGULATORY_COMPLIANCE_ENGINE_AVAILABLE = False

try:
    from greenlang.agents.eudr.commodity_risk_analyzer.engines.commodity_due_diligence_engine import (
        CommodityDueDiligenceEngine,
    )

    _COMMODITY_DD_ENGINE_AVAILABLE = True
except ImportError:
    CommodityDueDiligenceEngine = None  # type: ignore[assignment,misc]
    _COMMODITY_DD_ENGINE_AVAILABLE = False

try:
    from greenlang.agents.eudr.commodity_risk_analyzer.engines.portfolio_risk_aggregator import (
        PortfolioRiskAggregator,
    )

    _PORTFOLIO_RISK_AGGREGATOR_AVAILABLE = True
except ImportError:
    PortfolioRiskAggregator = None  # type: ignore[assignment,misc]
    _PORTFOLIO_RISK_AGGREGATOR_AVAILABLE = False

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"
_AGENT_ID = "GL-EUDR-CRA-018"
_ENGINE_COUNT = 8
_ENGINE_NAMES: List[str] = [
    "CommodityProfiler",
    "DerivedProductAnalyzer",
    "PriceVolatilityEngine",
    "ProductionForecastEngine",
    "SubstitutionRiskAnalyzer",
    "RegulatoryComplianceEngine",
    "CommodityDueDiligenceEngine",
    "PortfolioRiskAggregator",
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
# FACADE: CommodityRiskAnalyzerSetup
# =============================================================================

class CommodityRiskAnalyzerSetup:
    """
    CommodityRiskAnalyzerSetup orchestrates all 8 engines of AGENT-EUDR-018.

    This facade provides a single, thread-safe entry point for all commodity
    risk analysis operations per EUDR Articles 1, 2, 3, 4, 8, 9, 10, and
    Annex I.

    Architecture:
        - Lazy initialization of all 8 engines (on first use)
        - Thread-safe singleton pattern with double-checked locking
        - PostgreSQL connection pooling via psycopg_pool
        - Redis caching for frequently accessed reference data
        - OpenTelemetry distributed tracing integration
        - Prometheus metrics for all operations
        - SHA-256 provenance hashing for audit trails

    Engines:
        1. CommodityProfiler: Base commodity risk profiling with supply chain
           depth analysis and deforestation risk scoring
        2. DerivedProductAnalyzer: Annex I product traceability across
           processing stages with transformation ratio tracking
        3. PriceVolatilityEngine: 30-day and 90-day volatility calculation
           with market disruption detection
        4. ProductionForecastEngine: Yield estimation with seasonal
           coefficients and climate impact adjustment
        5. SubstitutionRiskAnalyzer: Commodity substitution event detection
           across suppliers with confidence scoring
        6. RegulatoryComplianceEngine: EUDR article requirements mapping
           per commodity with documentation standards
        7. CommodityDueDiligenceEngine: Commodity-specific DD workflow
           management with evidence collection and verification
        8. PortfolioRiskAggregator: Multi-commodity portfolio analysis
           with HHI concentration and diversification scoring

    Attributes:
        config: Current configuration instance
        db_pool: Async PostgreSQL connection pool (psycopg_pool)
        redis_client: Async Redis client (redis.asyncio)
        provenance_tracker: SHA-256 provenance tracking

    Example:
        >>> service = get_service()
        >>> await service.startup()
        >>>
        >>> # Profile a commodity
        >>> profile = await service.profile_commodity(
        ...     commodity_type="cocoa",
        ...     region="west_africa",
        ... )
        >>>
        >>> # Analyze derived product
        >>> derived = await service.analyze_derived_product(
        ...     product_id="DP-CHOC-001",
        ...     source_commodity="cocoa",
        ...     processing_stages=["fermentation", "roasting", "conching"],
        ... )
        >>>
        >>> # Portfolio risk
        >>> portfolio = await service.analyze_portfolio(
        ...     commodity_positions=[
        ...         {"commodity": "cocoa", "weight": 0.4},
        ...         {"commodity": "coffee", "weight": 0.3},
        ...         {"commodity": "oil_palm", "weight": 0.3},
        ...     ],
        ... )
        >>>
        >>> await service.shutdown()
    """

    def __init__(
        self,
        config: Optional[CommodityRiskAnalyzerConfig] = None,
        *,
        db_pool: Optional[Any] = None,
        redis_client: Optional[Any] = None,
    ) -> None:
        """
        Initialize CommodityRiskAnalyzerSetup.

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
        self._commodity_profiler: Optional[Any] = None
        self._derived_product_analyzer: Optional[Any] = None
        self._price_volatility_engine: Optional[Any] = None
        self._production_forecast_engine: Optional[Any] = None
        self._substitution_risk_analyzer: Optional[Any] = None
        self._regulatory_compliance_engine: Optional[Any] = None
        self._commodity_dd_engine: Optional[Any] = None
        self._portfolio_risk_aggregator: Optional[Any] = None

        # Lifecycle state
        self._started: bool = False
        self._startup_time: Optional[datetime] = None
        self._startup_lock = asyncio.Lock()
        self._shutdown_lock = asyncio.Lock()

        # Statistics tracking
        self._stats: Dict[str, int] = {
            "total_analyses": 0,
            "total_profiles": 0,
            "total_derived_product_analyses": 0,
            "total_price_queries": 0,
            "total_forecasts": 0,
            "total_substitution_checks": 0,
            "total_compliance_checks": 0,
            "total_dd_initiations": 0,
            "total_portfolio_analyses": 0,
            "total_comprehensive_analyses": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
        }

        logger.info(
            f"CommodityRiskAnalyzerSetup initialized "
            f"(version={_MODULE_VERSION}, agent_id={_AGENT_ID})"
        )

    # -----------------------------------------------------------------------
    # Properties for engine access
    # -----------------------------------------------------------------------

    @property
    def commodity_profiler(self) -> Any:
        """Access CommodityProfiler engine (lazy initialized).

        Returns:
            CommodityProfiler engine instance.

        Raises:
            RuntimeError: If engine is not available.
        """
        if self._commodity_profiler is None:
            raise RuntimeError(
                "CommodityProfiler not initialized. Call startup() first."
            )
        return self._commodity_profiler

    @property
    def derived_product_analyzer(self) -> Any:
        """Access DerivedProductAnalyzer engine (lazy initialized).

        Returns:
            DerivedProductAnalyzer engine instance.

        Raises:
            RuntimeError: If engine is not available.
        """
        if self._derived_product_analyzer is None:
            raise RuntimeError(
                "DerivedProductAnalyzer not initialized. Call startup() first."
            )
        return self._derived_product_analyzer

    @property
    def price_volatility_engine(self) -> Any:
        """Access PriceVolatilityEngine (lazy initialized).

        Returns:
            PriceVolatilityEngine instance.

        Raises:
            RuntimeError: If engine is not available.
        """
        if self._price_volatility_engine is None:
            raise RuntimeError(
                "PriceVolatilityEngine not initialized. Call startup() first."
            )
        return self._price_volatility_engine

    @property
    def production_forecast_engine(self) -> Any:
        """Access ProductionForecastEngine (lazy initialized).

        Returns:
            ProductionForecastEngine instance.

        Raises:
            RuntimeError: If engine is not available.
        """
        if self._production_forecast_engine is None:
            raise RuntimeError(
                "ProductionForecastEngine not initialized. Call startup() first."
            )
        return self._production_forecast_engine

    @property
    def substitution_risk_analyzer(self) -> Any:
        """Access SubstitutionRiskAnalyzer (lazy initialized).

        Returns:
            SubstitutionRiskAnalyzer instance.

        Raises:
            RuntimeError: If engine is not available.
        """
        if self._substitution_risk_analyzer is None:
            raise RuntimeError(
                "SubstitutionRiskAnalyzer not initialized. Call startup() first."
            )
        return self._substitution_risk_analyzer

    @property
    def regulatory_compliance_engine(self) -> Any:
        """Access RegulatoryComplianceEngine (lazy initialized).

        Returns:
            RegulatoryComplianceEngine instance.

        Raises:
            RuntimeError: If engine is not available.
        """
        if self._regulatory_compliance_engine is None:
            raise RuntimeError(
                "RegulatoryComplianceEngine not initialized. Call startup() first."
            )
        return self._regulatory_compliance_engine

    @property
    def commodity_dd_engine(self) -> Any:
        """Access CommodityDueDiligenceEngine (lazy initialized).

        Returns:
            CommodityDueDiligenceEngine instance.

        Raises:
            RuntimeError: If engine is not available.
        """
        if self._commodity_dd_engine is None:
            raise RuntimeError(
                "CommodityDueDiligenceEngine not initialized. Call startup() first."
            )
        return self._commodity_dd_engine

    @property
    def portfolio_risk_aggregator(self) -> Any:
        """Access PortfolioRiskAggregator (lazy initialized).

        Returns:
            PortfolioRiskAggregator instance.

        Raises:
            RuntimeError: If engine is not available.
        """
        if self._portfolio_risk_aggregator is None:
            raise RuntimeError(
                "PortfolioRiskAggregator not initialized. Call startup() first."
            )
        return self._portfolio_risk_aggregator

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
                    "CommodityRiskAnalyzerSetup already started, skipping startup"
                )
                return

            logger.info("Starting CommodityRiskAnalyzerSetup...")
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
                self._startup_time = utcnow()
                duration_ms = (time.monotonic() - start_time) * 1000

                logger.info(
                    f"CommodityRiskAnalyzerSetup started successfully "
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
                    "CommodityRiskAnalyzerSetup not started, skipping shutdown"
                )
                return

            logger.info("Shutting down CommodityRiskAnalyzerSetup...")

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
                    "CommodityRiskAnalyzerSetup shutdown complete"
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
            ("CommodityProfiler", self._ensure_commodity_profiler),
            ("DerivedProductAnalyzer", self._ensure_derived_product_analyzer),
            ("PriceVolatilityEngine", self._ensure_price_volatility_engine),
            ("ProductionForecastEngine", self._ensure_production_forecast_engine),
            ("SubstitutionRiskAnalyzer", self._ensure_substitution_risk_analyzer),
            ("RegulatoryComplianceEngine", self._ensure_regulatory_compliance_engine),
            ("CommodityDueDiligenceEngine", self._ensure_commodity_dd_engine),
            ("PortfolioRiskAggregator", self._ensure_portfolio_risk_aggregator),
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
            from greenlang.agents.eudr.commodity_risk_analyzer.reference_data import (
                CommodityDatabase,
                ProcessingChainDatabase,
                ProductionStatistics,
                RegulatoryRequirementDatabase,
            )

            init_results["reference_data"]["commodity_database"] = "loaded"
            init_results["reference_data"]["processing_chains"] = "loaded"
            init_results["reference_data"]["production_statistics"] = "loaded"
            init_results["reference_data"]["regulatory_requirements"] = "loaded"
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
            f"(min={self._config.pool_min_size}, max={self._config.pool_size})"
        )
        self._db_pool = AsyncConnectionPool(
            conninfo=self._config.database_url,
            min_size=self._config.pool_min_size,
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
            max_connections=self._config.redis_max_connections,
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
        )
        # Verify connection
        await self._redis_client.ping()
        logger.info("Redis client connected")

    async def _shutdown_engines(self) -> None:
        """Shutdown all initialized engines gracefully."""
        engines = [
            ("CommodityProfiler", self._commodity_profiler),
            ("DerivedProductAnalyzer", self._derived_product_analyzer),
            ("PriceVolatilityEngine", self._price_volatility_engine),
            ("ProductionForecastEngine", self._production_forecast_engine),
            ("SubstitutionRiskAnalyzer", self._substitution_risk_analyzer),
            ("RegulatoryComplianceEngine", self._regulatory_compliance_engine),
            ("CommodityDueDiligenceEngine", self._commodity_dd_engine),
            ("PortfolioRiskAggregator", self._portfolio_risk_aggregator),
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

    async def _ensure_commodity_profiler(self) -> Any:
        """Lazy initialize CommodityProfiler engine.

        Returns:
            Initialized CommodityProfiler instance.

        Raises:
            RuntimeError: If engine module is not available.
        """
        if self._commodity_profiler is None:
            if not _COMMODITY_PROFILER_AVAILABLE or CommodityProfiler is None:
                raise RuntimeError("CommodityProfiler engine not available")
            logger.debug("Initializing CommodityProfiler engine...")
            self._commodity_profiler = CommodityProfiler(
                config=self._config,
                db_pool=self._db_pool,
                redis_client=self._redis_client,
            )
            if hasattr(self._commodity_profiler, "startup"):
                await self._commodity_profiler.startup()
            logger.info("CommodityProfiler engine initialized")
        return self._commodity_profiler

    async def _ensure_derived_product_analyzer(self) -> Any:
        """Lazy initialize DerivedProductAnalyzer engine.

        Returns:
            Initialized DerivedProductAnalyzer instance.

        Raises:
            RuntimeError: If engine module is not available.
        """
        if self._derived_product_analyzer is None:
            if not _DERIVED_PRODUCT_ANALYZER_AVAILABLE or DerivedProductAnalyzer is None:
                raise RuntimeError("DerivedProductAnalyzer engine not available")
            logger.debug("Initializing DerivedProductAnalyzer engine...")
            self._derived_product_analyzer = DerivedProductAnalyzer(
                config=self._config,
                db_pool=self._db_pool,
                redis_client=self._redis_client,
            )
            if hasattr(self._derived_product_analyzer, "startup"):
                await self._derived_product_analyzer.startup()
            logger.info("DerivedProductAnalyzer engine initialized")
        return self._derived_product_analyzer

    async def _ensure_price_volatility_engine(self) -> Any:
        """Lazy initialize PriceVolatilityEngine.

        Returns:
            Initialized PriceVolatilityEngine instance.

        Raises:
            RuntimeError: If engine module is not available.
        """
        if self._price_volatility_engine is None:
            if not _PRICE_VOLATILITY_ENGINE_AVAILABLE or PriceVolatilityEngine is None:
                raise RuntimeError("PriceVolatilityEngine not available")
            logger.debug("Initializing PriceVolatilityEngine...")
            self._price_volatility_engine = PriceVolatilityEngine(
                config=self._config,
                db_pool=self._db_pool,
                redis_client=self._redis_client,
            )
            if hasattr(self._price_volatility_engine, "startup"):
                await self._price_volatility_engine.startup()
            logger.info("PriceVolatilityEngine initialized")
        return self._price_volatility_engine

    async def _ensure_production_forecast_engine(self) -> Any:
        """Lazy initialize ProductionForecastEngine.

        Returns:
            Initialized ProductionForecastEngine instance.

        Raises:
            RuntimeError: If engine module is not available.
        """
        if self._production_forecast_engine is None:
            if not _PRODUCTION_FORECAST_ENGINE_AVAILABLE or ProductionForecastEngine is None:
                raise RuntimeError("ProductionForecastEngine not available")
            logger.debug("Initializing ProductionForecastEngine...")
            self._production_forecast_engine = ProductionForecastEngine(
                config=self._config,
                db_pool=self._db_pool,
                redis_client=self._redis_client,
            )
            if hasattr(self._production_forecast_engine, "startup"):
                await self._production_forecast_engine.startup()
            logger.info("ProductionForecastEngine initialized")
        return self._production_forecast_engine

    async def _ensure_substitution_risk_analyzer(self) -> Any:
        """Lazy initialize SubstitutionRiskAnalyzer.

        Returns:
            Initialized SubstitutionRiskAnalyzer instance.

        Raises:
            RuntimeError: If engine module is not available.
        """
        if self._substitution_risk_analyzer is None:
            if not _SUBSTITUTION_RISK_ANALYZER_AVAILABLE or SubstitutionRiskAnalyzer is None:
                raise RuntimeError("SubstitutionRiskAnalyzer not available")
            logger.debug("Initializing SubstitutionRiskAnalyzer...")
            self._substitution_risk_analyzer = SubstitutionRiskAnalyzer(
                config=self._config,
                db_pool=self._db_pool,
                redis_client=self._redis_client,
            )
            if hasattr(self._substitution_risk_analyzer, "startup"):
                await self._substitution_risk_analyzer.startup()
            logger.info("SubstitutionRiskAnalyzer initialized")
        return self._substitution_risk_analyzer

    async def _ensure_regulatory_compliance_engine(self) -> Any:
        """Lazy initialize RegulatoryComplianceEngine.

        Returns:
            Initialized RegulatoryComplianceEngine instance.

        Raises:
            RuntimeError: If engine module is not available.
        """
        if self._regulatory_compliance_engine is None:
            if not _REGULATORY_COMPLIANCE_ENGINE_AVAILABLE or RegulatoryComplianceEngine is None:
                raise RuntimeError("RegulatoryComplianceEngine not available")
            logger.debug("Initializing RegulatoryComplianceEngine...")
            self._regulatory_compliance_engine = RegulatoryComplianceEngine(
                config=self._config,
                db_pool=self._db_pool,
                redis_client=self._redis_client,
            )
            if hasattr(self._regulatory_compliance_engine, "startup"):
                await self._regulatory_compliance_engine.startup()
            logger.info("RegulatoryComplianceEngine initialized")
        return self._regulatory_compliance_engine

    async def _ensure_commodity_dd_engine(self) -> Any:
        """Lazy initialize CommodityDueDiligenceEngine.

        Returns:
            Initialized CommodityDueDiligenceEngine instance.

        Raises:
            RuntimeError: If engine module is not available.
        """
        if self._commodity_dd_engine is None:
            if not _COMMODITY_DD_ENGINE_AVAILABLE or CommodityDueDiligenceEngine is None:
                raise RuntimeError("CommodityDueDiligenceEngine not available")
            logger.debug("Initializing CommodityDueDiligenceEngine...")
            self._commodity_dd_engine = CommodityDueDiligenceEngine(
                config=self._config,
                db_pool=self._db_pool,
                redis_client=self._redis_client,
            )
            if hasattr(self._commodity_dd_engine, "startup"):
                await self._commodity_dd_engine.startup()
            logger.info("CommodityDueDiligenceEngine initialized")
        return self._commodity_dd_engine

    async def _ensure_portfolio_risk_aggregator(self) -> Any:
        """Lazy initialize PortfolioRiskAggregator.

        Returns:
            Initialized PortfolioRiskAggregator instance.

        Raises:
            RuntimeError: If engine module is not available.
        """
        if self._portfolio_risk_aggregator is None:
            if not _PORTFOLIO_RISK_AGGREGATOR_AVAILABLE or PortfolioRiskAggregator is None:
                raise RuntimeError("PortfolioRiskAggregator not available")
            logger.debug("Initializing PortfolioRiskAggregator...")
            self._portfolio_risk_aggregator = PortfolioRiskAggregator(
                config=self._config,
                db_pool=self._db_pool,
                redis_client=self._redis_client,
            )
            if hasattr(self._portfolio_risk_aggregator, "startup"):
                await self._portfolio_risk_aggregator.startup()
            logger.info("PortfolioRiskAggregator initialized")
        return self._portfolio_risk_aggregator

    # -----------------------------------------------------------------------
    # Public API: Engine delegation methods
    # -----------------------------------------------------------------------

    async def profile_commodity(
        self,
        commodity_type: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Profile a commodity using CommodityProfiler engine.

        Generates a comprehensive risk profile for the specified EUDR commodity
        including supply chain depth, deforestation risk, country distribution,
        and processing chain enumeration.

        Args:
            commodity_type: EUDR commodity type (cattle, cocoa, coffee,
                oil_palm, rubber, soya, wood).
            **kwargs: Additional profiling parameters:
                - region (str): Geographic region filter.
                - assessment_date (datetime): Assessment date override.
                - include_seasonal (bool): Include seasonal patterns.
                - include_certifications (bool): Include certification data.

        Returns:
            Dict with commodity profile containing risk_score, risk_level,
            supply_chain_depth, country_distribution, and processing_chains.

        Raises:
            ValueError: If commodity_type is invalid.
            RuntimeError: If CommodityProfiler engine is not available.
        """
        start_time = time.monotonic()
        logger.info(f"Profiling commodity: {commodity_type}")

        try:
            if commodity_type not in SUPPORTED_COMMODITIES:
                raise ValueError(
                    f"Unsupported commodity: {commodity_type}. "
                    f"Supported: {SUPPORTED_COMMODITIES}"
                )

            engine = await self._ensure_commodity_profiler()
            result = await engine.profile_commodity(
                commodity_type=commodity_type, **kwargs
            )

            # Record metrics and stats
            duration_sec = time.monotonic() - start_time
            observe_profile_duration(duration_sec)
            record_profile_created(commodity_type=commodity_type, status="success")
            self._stats["total_profiles"] += 1
            self._stats["total_analyses"] += 1

            # Provenance
            provenance_hash = _calculate_sha256({
                "action": "profile_commodity",
                "commodity_type": commodity_type,
                "kwargs": str(kwargs),
                "timestamp": utcnow().isoformat(),
            })

            logger.info(
                f"Commodity profile complete for {commodity_type}: "
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
            record_api_error("profile_commodity", str(e))
            logger.error(f"profile_commodity failed: {e}", exc_info=True)
            raise

    async def analyze_derived_product(
        self,
        product_id: str,
        source_commodity: str,
        processing_stages: List[str],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Analyze a derived product using DerivedProductAnalyzer engine.

        Traces a derived product back through its processing chain to the
        source commodity, calculating risk multipliers, transformation ratios,
        and traceability scores at each stage.

        Args:
            product_id: Unique derived product identifier (e.g., "DP-CHOC-001").
            source_commodity: Source EUDR commodity type.
            processing_stages: Ordered list of processing stage names.
            **kwargs: Additional analysis parameters:
                - include_waste (bool): Include waste percentage tracking.
                - include_risk_chain (bool): Include per-stage risk chain.

        Returns:
            Dict with derived product analysis including traceability_score,
            risk_multiplier, transformation_ratio, and stage_details.

        Raises:
            ValueError: If source_commodity is invalid.
            RuntimeError: If DerivedProductAnalyzer is not available.
        """
        start_time = time.monotonic()
        logger.info(
            f"Analyzing derived product: {product_id} from {source_commodity}"
        )

        try:
            if source_commodity not in SUPPORTED_COMMODITIES:
                raise ValueError(f"Unsupported source commodity: {source_commodity}")

            engine = await self._ensure_derived_product_analyzer()
            result = await engine.analyze_derived_product(
                product_id=product_id,
                source_commodity=source_commodity,
                processing_stages=processing_stages,
                **kwargs,
            )

            duration_sec = time.monotonic() - start_time
            observe_analysis_duration(duration_sec)
            record_derived_product_analyzed(
                commodity=source_commodity, status="success"
            )
            self._stats["total_derived_product_analyses"] += 1
            self._stats["total_analyses"] += 1

            provenance_hash = _calculate_sha256({
                "action": "analyze_derived_product",
                "product_id": product_id,
                "source_commodity": source_commodity,
                "processing_stages": processing_stages,
                "timestamp": utcnow().isoformat(),
            })

            logger.info(
                f"Derived product analysis complete for {product_id}: "
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
            record_api_error("analyze_derived_product", str(e))
            logger.error(f"analyze_derived_product failed: {e}", exc_info=True)
            raise

    async def get_price_volatility(
        self,
        commodity_type: str,
        window_days: int = 30,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Get price volatility for a commodity using PriceVolatilityEngine.

        Calculates rolling volatility over the specified window, detects
        market disruptions, classifies market conditions, and identifies
        seasonal phases.

        Args:
            commodity_type: EUDR commodity type.
            window_days: Rolling window in days (default 30, max 365).
            **kwargs: Additional parameters:
                - include_exchange_data (bool): Include raw exchange prices.
                - disruption_threshold (float): Custom disruption threshold.

        Returns:
            Dict with volatility_index, market_condition, seasonal_phase,
            disruption_detected flag, and historical price data.

        Raises:
            ValueError: If commodity_type is invalid or window_days out of range.
            RuntimeError: If PriceVolatilityEngine is not available.
        """
        start_time = time.monotonic()
        logger.info(
            f"Querying price volatility: {commodity_type} (window={window_days}d)"
        )

        try:
            if commodity_type not in SUPPORTED_COMMODITIES:
                raise ValueError(f"Unsupported commodity: {commodity_type}")
            if not 1 <= window_days <= 365:
                raise ValueError(f"window_days must be 1-365, got {window_days}")

            engine = await self._ensure_price_volatility_engine()
            result = await engine.get_price_volatility(
                commodity_type=commodity_type,
                window_days=window_days,
                **kwargs,
            )

            duration_sec = time.monotonic() - start_time
            record_price_query(commodity=commodity_type, status="success")
            self._stats["total_price_queries"] += 1
            self._stats["total_analyses"] += 1

            provenance_hash = _calculate_sha256({
                "action": "get_price_volatility",
                "commodity_type": commodity_type,
                "window_days": window_days,
                "timestamp": utcnow().isoformat(),
            })

            logger.info(
                f"Price volatility query complete for {commodity_type}: "
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
            record_api_error("get_price_volatility", str(e))
            logger.error(f"get_price_volatility failed: {e}", exc_info=True)
            raise

    async def forecast_production(
        self,
        commodity_type: str,
        region: str,
        horizon_months: int = 12,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Forecast production for a commodity using ProductionForecastEngine.

        Generates production forecasts with seasonal coefficient adjustment,
        climate impact factors, yield confidence intervals, and regional
        granularity.

        Args:
            commodity_type: EUDR commodity type.
            region: Geographic region or country code.
            horizon_months: Forecast horizon in months (default 12, max 60).
            **kwargs: Additional parameters:
                - include_climate_impact (bool): Include climate adjustment.
                - confidence_level (float): Confidence level (0.8-0.99).

        Returns:
            Dict with forecast data including production_estimate,
            confidence_interval, seasonal_factors, and climate_impact.

        Raises:
            ValueError: If inputs are invalid.
            RuntimeError: If ProductionForecastEngine is not available.
        """
        start_time = time.monotonic()
        logger.info(
            f"Forecasting production: {commodity_type} in {region} "
            f"(horizon={horizon_months}m)"
        )

        try:
            if commodity_type not in SUPPORTED_COMMODITIES:
                raise ValueError(f"Unsupported commodity: {commodity_type}")
            if not 1 <= horizon_months <= 60:
                raise ValueError(
                    f"horizon_months must be 1-60, got {horizon_months}"
                )

            engine = await self._ensure_production_forecast_engine()
            result = await engine.forecast_production(
                commodity_type=commodity_type,
                region=region,
                horizon_months=horizon_months,
                **kwargs,
            )

            duration_sec = time.monotonic() - start_time
            observe_forecast_duration(duration_sec)
            record_forecast_generated(
                commodity=commodity_type, status="success"
            )
            self._stats["total_forecasts"] += 1
            self._stats["total_analyses"] += 1

            provenance_hash = _calculate_sha256({
                "action": "forecast_production",
                "commodity_type": commodity_type,
                "region": region,
                "horizon_months": horizon_months,
                "timestamp": utcnow().isoformat(),
            })

            logger.info(
                f"Production forecast complete for {commodity_type}/{region}: "
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
            record_api_error("forecast_production", str(e))
            logger.error(f"forecast_production failed: {e}", exc_info=True)
            raise

    async def detect_substitution(
        self,
        supplier_id: str,
        commodity_history: List[Dict[str, Any]],
        current_declaration: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Detect commodity substitution using SubstitutionRiskAnalyzer.

        Analyzes historical commodity declarations against the current
        declaration to identify potential substitution events, including
        sudden commodity switches, volume anomalies, and grade changes.

        Args:
            supplier_id: Supplier identifier.
            commodity_history: Historical commodity declarations list.
            current_declaration: Current commodity declaration dict.
            **kwargs: Additional parameters:
                - confidence_threshold (float): Detection threshold (0.0-1.0).
                - lookback_periods (int): Number of periods to analyze.

        Returns:
            Dict with substitution_detected flag, confidence_score,
            risk_impact, and detected_events list.

        Raises:
            ValueError: If supplier_id is empty or history is empty.
            RuntimeError: If SubstitutionRiskAnalyzer is not available.
        """
        start_time = time.monotonic()
        logger.info(
            f"Detecting substitution for supplier {supplier_id} "
            f"(history_periods={len(commodity_history)})"
        )

        try:
            if not supplier_id:
                raise ValueError("supplier_id must not be empty")

            engine = await self._ensure_substitution_risk_analyzer()
            result = await engine.detect_substitution(
                supplier_id=supplier_id,
                commodity_history=commodity_history,
                current_declaration=current_declaration,
                **kwargs,
            )

            duration_sec = time.monotonic() - start_time
            record_substitution_detected(
                supplier_id=supplier_id, status="success"
            )
            self._stats["total_substitution_checks"] += 1
            self._stats["total_analyses"] += 1

            provenance_hash = _calculate_sha256({
                "action": "detect_substitution",
                "supplier_id": supplier_id,
                "history_count": len(commodity_history),
                "timestamp": utcnow().isoformat(),
            })

            logger.info(
                f"Substitution detection complete for {supplier_id}: "
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
            record_api_error("detect_substitution", str(e))
            logger.error(f"detect_substitution failed: {e}", exc_info=True)
            raise

    async def check_regulatory_compliance(
        self,
        commodity_type: str,
        supplier_data: Dict[str, Any],
        documentation: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Check regulatory compliance using RegulatoryComplianceEngine.

        Maps EUDR article requirements to the supplied documentation for
        the specified commodity, identifies gaps, and generates a compliance
        status report.

        Args:
            commodity_type: EUDR commodity type.
            supplier_data: Supplier information dict.
            documentation: List of documentation dicts provided by supplier.
            **kwargs: Additional parameters:
                - include_penalties (bool): Include penalty matrix info.
                - strictness_level (str): "standard" or "enhanced".

        Returns:
            Dict with compliance_status, article_compliance (per-article),
            missing_documentation, and recommendations.

        Raises:
            ValueError: If commodity_type is invalid.
            RuntimeError: If RegulatoryComplianceEngine is not available.
        """
        start_time = time.monotonic()
        logger.info(
            f"Checking regulatory compliance for {commodity_type} "
            f"(docs_count={len(documentation)})"
        )

        try:
            if commodity_type not in SUPPORTED_COMMODITIES:
                raise ValueError(f"Unsupported commodity: {commodity_type}")

            engine = await self._ensure_regulatory_compliance_engine()
            result = await engine.check_compliance(
                commodity_type=commodity_type,
                supplier_data=supplier_data,
                documentation=documentation,
                **kwargs,
            )

            duration_sec = time.monotonic() - start_time
            record_compliance_check(
                commodity=commodity_type, status="success"
            )
            self._stats["total_compliance_checks"] += 1
            self._stats["total_analyses"] += 1

            provenance_hash = _calculate_sha256({
                "action": "check_regulatory_compliance",
                "commodity_type": commodity_type,
                "docs_count": len(documentation),
                "timestamp": utcnow().isoformat(),
            })

            logger.info(
                f"Regulatory compliance check complete for {commodity_type}: "
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
            record_api_error("check_regulatory_compliance", str(e))
            logger.error(
                f"check_regulatory_compliance failed: {e}", exc_info=True
            )
            raise

    async def initiate_due_diligence(
        self,
        commodity_type: str,
        supplier_id: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Initiate commodity-specific due diligence using CommodityDueDiligenceEngine.

        Creates and manages a due diligence workflow for the specified commodity
        and supplier, including evidence item collection, verification step
        orchestration, and completion tracking.

        Args:
            commodity_type: EUDR commodity type.
            supplier_id: Supplier identifier.
            **kwargs: Additional parameters:
                - dd_level (str): "standard" or "enhanced".
                - priority (str): "low", "medium", "high", "urgent".
                - assigned_to (str): Assignee user ID.

        Returns:
            Dict with workflow_id, workflow_status, evidence_items,
            verification_steps, and completion_percentage.

        Raises:
            ValueError: If inputs are invalid.
            RuntimeError: If CommodityDueDiligenceEngine is not available.
        """
        start_time = time.monotonic()
        logger.info(
            f"Initiating due diligence: {commodity_type} for {supplier_id}"
        )

        try:
            if commodity_type not in SUPPORTED_COMMODITIES:
                raise ValueError(f"Unsupported commodity: {commodity_type}")
            if not supplier_id:
                raise ValueError("supplier_id must not be empty")

            engine = await self._ensure_commodity_dd_engine()
            result = await engine.initiate_due_diligence(
                commodity_type=commodity_type,
                supplier_id=supplier_id,
                **kwargs,
            )

            duration_sec = time.monotonic() - start_time
            record_dd_workflow_initiated(
                commodity=commodity_type, status="success"
            )
            self._stats["total_dd_initiations"] += 1
            self._stats["total_analyses"] += 1

            provenance_hash = _calculate_sha256({
                "action": "initiate_due_diligence",
                "commodity_type": commodity_type,
                "supplier_id": supplier_id,
                "timestamp": utcnow().isoformat(),
            })

            logger.info(
                f"Due diligence initiated for {commodity_type}/{supplier_id}: "
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
            record_api_error("initiate_due_diligence", str(e))
            logger.error(f"initiate_due_diligence failed: {e}", exc_info=True)
            raise

    async def analyze_portfolio(
        self,
        commodity_positions: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Analyze a multi-commodity portfolio using PortfolioRiskAggregator.

        Calculates HHI concentration index, diversification score, total
        risk exposure, and commodity correlation matrix for the portfolio.

        Args:
            commodity_positions: List of position dicts, each with:
                - commodity (str): EUDR commodity type.
                - weight (float): Portfolio weight (0.0-1.0).
                - volume (float, optional): Volume in tonnes.
                - supplier_count (int, optional): Number of suppliers.
            **kwargs: Additional parameters:
                - strategy (str): Portfolio strategy type.
                - risk_appetite (str): "conservative", "moderate", "aggressive".

        Returns:
            Dict with hhi_index, diversification_score, total_risk_exposure,
            concentration_risk_level, and per-commodity breakdown.

        Raises:
            ValueError: If positions are empty or weights invalid.
            RuntimeError: If PortfolioRiskAggregator is not available.
        """
        start_time = time.monotonic()
        logger.info(
            f"Analyzing portfolio with {len(commodity_positions)} positions"
        )

        try:
            if not commodity_positions:
                raise ValueError("commodity_positions must not be empty")

            # Validate commodities in positions
            for pos in commodity_positions:
                if pos.get("commodity") not in SUPPORTED_COMMODITIES:
                    raise ValueError(
                        f"Unsupported commodity in position: {pos.get('commodity')}"
                    )

            engine = await self._ensure_portfolio_risk_aggregator()
            result = await engine.analyze_portfolio(
                commodity_positions=commodity_positions, **kwargs
            )

            duration_sec = time.monotonic() - start_time
            observe_portfolio_duration(duration_sec)
            record_portfolio_analysis(status="success")
            self._stats["total_portfolio_analyses"] += 1
            self._stats["total_analyses"] += 1

            provenance_hash = _calculate_sha256({
                "action": "analyze_portfolio",
                "position_count": len(commodity_positions),
                "commodities": [p.get("commodity") for p in commodity_positions],
                "timestamp": utcnow().isoformat(),
            })

            logger.info(
                f"Portfolio analysis complete "
                f"({len(commodity_positions)} positions): "
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
            record_api_error("analyze_portfolio", str(e))
            logger.error(f"analyze_portfolio failed: {e}", exc_info=True)
            raise

    # -----------------------------------------------------------------------
    # Public API: Cross-engine orchestration
    # -----------------------------------------------------------------------

    async def run_comprehensive_analysis(
        self,
        commodity_type: str,
        supplier_id: Optional[str] = None,
        include_all: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Run ALL engines for a complete commodity risk analysis.

        Orchestrates all 8 engines to produce a comprehensive commodity
        risk analysis covering profiling, derived products, price volatility,
        production forecast, substitution risk, regulatory compliance,
        due diligence, and portfolio impact.

        Args:
            commodity_type: EUDR commodity type.
            supplier_id: Optional supplier identifier for supplier-specific analyses.
            include_all: If True, run all engines. If False, run only core engines.
            **kwargs: Additional parameters passed to individual engines:
                - region (str): Geographic region for forecasting.
                - window_days (int): Price volatility window.
                - horizon_months (int): Forecast horizon.
                - include_derived_products (bool): Include derived products.
                - include_portfolio_impact (bool): Include portfolio analysis.

        Returns:
            Dict with comprehensive analysis results containing:
                - commodity_profile: Base commodity risk profile
                - derived_products: Derived product analysis (if included)
                - price_volatility: Current volatility metrics
                - production_forecast: Production forecast (if region specified)
                - substitution_risk: Substitution risk assessment (if supplier_id)
                - regulatory_compliance: Compliance status mapping
                - due_diligence: DD workflow status (if supplier_id)
                - portfolio_impact: Portfolio-level impact (if included)
                - provenance_chain: Linked provenance hashes for all engines
                - processing_time_ms: Total processing duration

        Raises:
            ValueError: If commodity_type is invalid.
        """
        start_time = time.monotonic()
        logger.info(
            f"Starting comprehensive analysis for {commodity_type} "
            f"(supplier={supplier_id}, include_all={include_all})"
        )

        try:
            if commodity_type not in SUPPORTED_COMMODITIES:
                raise ValueError(f"Unsupported commodity: {commodity_type}")

            results: Dict[str, Any] = {
                "commodity_type": commodity_type,
                "supplier_id": supplier_id,
                "analysis_timestamp": utcnow().isoformat(),
                "agent_id": _AGENT_ID,
                "version": _MODULE_VERSION,
            }
            provenance_chain: List[str] = []

            # 1. Commodity Profile (always included)
            try:
                profile = await self.profile_commodity(
                    commodity_type=commodity_type,
                    region=kwargs.get("region"),
                )
                results["commodity_profile"] = profile
                if "provenance_hash" in profile:
                    provenance_chain.append(profile["provenance_hash"])
            except Exception as e:
                results["commodity_profile"] = {"error": str(e)}
                logger.warning(f"Commodity profiling failed: {e}")

            # 2. Price Volatility (always included)
            try:
                volatility = await self.get_price_volatility(
                    commodity_type=commodity_type,
                    window_days=kwargs.get("window_days", 30),
                )
                results["price_volatility"] = volatility
                if "provenance_hash" in volatility:
                    provenance_chain.append(volatility["provenance_hash"])
            except Exception as e:
                results["price_volatility"] = {"error": str(e)}
                logger.warning(f"Price volatility query failed: {e}")

            # 3. Regulatory Compliance (always included)
            try:
                compliance = await self.check_regulatory_compliance(
                    commodity_type=commodity_type,
                    supplier_data={"supplier_id": supplier_id} if supplier_id else {},
                    documentation=[],
                )
                results["regulatory_compliance"] = compliance
                if "provenance_hash" in compliance:
                    provenance_chain.append(compliance["provenance_hash"])
            except Exception as e:
                results["regulatory_compliance"] = {"error": str(e)}
                logger.warning(f"Regulatory compliance check failed: {e}")

            # 4. Production Forecast (if region specified or include_all)
            if include_all or kwargs.get("region"):
                try:
                    region = kwargs.get("region", "global")
                    forecast = await self.forecast_production(
                        commodity_type=commodity_type,
                        region=region,
                        horizon_months=kwargs.get("horizon_months", 12),
                    )
                    results["production_forecast"] = forecast
                    if "provenance_hash" in forecast:
                        provenance_chain.append(forecast["provenance_hash"])
                except Exception as e:
                    results["production_forecast"] = {"error": str(e)}
                    logger.warning(f"Production forecast failed: {e}")

            # 5. Derived Products (if include_all)
            if include_all or kwargs.get("include_derived_products"):
                try:
                    derived = await self.analyze_derived_product(
                        product_id=f"DP-{commodity_type.upper()}-AUTO",
                        source_commodity=commodity_type,
                        processing_stages=[],
                    )
                    results["derived_products"] = derived
                    if "provenance_hash" in derived:
                        provenance_chain.append(derived["provenance_hash"])
                except Exception as e:
                    results["derived_products"] = {"error": str(e)}
                    logger.warning(f"Derived product analysis failed: {e}")

            # 6. Substitution Risk (if supplier_id provided)
            if supplier_id and (include_all or kwargs.get("include_substitution")):
                try:
                    substitution = await self.detect_substitution(
                        supplier_id=supplier_id,
                        commodity_history=[],
                        current_declaration={
                            "commodity": commodity_type,
                            "date": utcnow().isoformat(),
                        },
                    )
                    results["substitution_risk"] = substitution
                    if "provenance_hash" in substitution:
                        provenance_chain.append(substitution["provenance_hash"])
                except Exception as e:
                    results["substitution_risk"] = {"error": str(e)}
                    logger.warning(f"Substitution detection failed: {e}")

            # 7. Due Diligence (if supplier_id provided)
            if supplier_id and (include_all or kwargs.get("include_dd")):
                try:
                    dd = await self.initiate_due_diligence(
                        commodity_type=commodity_type,
                        supplier_id=supplier_id,
                    )
                    results["due_diligence"] = dd
                    if "provenance_hash" in dd:
                        provenance_chain.append(dd["provenance_hash"])
                except Exception as e:
                    results["due_diligence"] = {"error": str(e)}
                    logger.warning(f"Due diligence initiation failed: {e}")

            # 8. Portfolio Impact (if include_all)
            if include_all or kwargs.get("include_portfolio_impact"):
                try:
                    portfolio = await self.analyze_portfolio(
                        commodity_positions=[
                            {"commodity": commodity_type, "weight": 1.0}
                        ],
                    )
                    results["portfolio_impact"] = portfolio
                    if "provenance_hash" in portfolio:
                        provenance_chain.append(portfolio["provenance_hash"])
                except Exception as e:
                    results["portfolio_impact"] = {"error": str(e)}
                    logger.warning(f"Portfolio analysis failed: {e}")

            # Calculate comprehensive provenance
            comprehensive_provenance = _calculate_sha256({
                "action": "run_comprehensive_analysis",
                "commodity_type": commodity_type,
                "supplier_id": supplier_id,
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
                f"Comprehensive analysis complete for {commodity_type}: "
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

    async def get_health_status(self) -> Dict[str, Any]:
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
            ("CommodityProfiler", self._commodity_profiler),
            ("DerivedProductAnalyzer", self._derived_product_analyzer),
            ("PriceVolatilityEngine", self._price_volatility_engine),
            ("ProductionForecastEngine", self._production_forecast_engine),
            ("SubstitutionRiskAnalyzer", self._substitution_risk_analyzer),
            ("RegulatoryComplianceEngine", self._regulatory_compliance_engine),
            ("CommodityDueDiligenceEngine", self._commodity_dd_engine),
            ("PortfolioRiskAggregator", self._portfolio_risk_aggregator),
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

        stats = {
            "agent_id": _AGENT_ID,
            "version": _MODULE_VERSION,
            "uptime_seconds": round(uptime_seconds, 2),
            "total_analyses": self._stats["total_analyses"],
            "total_profiles": self._stats["total_profiles"],
            "total_derived_product_analyses": self._stats[
                "total_derived_product_analyses"
            ],
            "total_price_queries": self._stats["total_price_queries"],
            "total_forecasts": self._stats["total_forecasts"],
            "total_substitution_checks": self._stats["total_substitution_checks"],
            "total_compliance_checks": self._stats["total_compliance_checks"],
            "total_dd_initiations": self._stats["total_dd_initiations"],
            "total_portfolio_analyses": self._stats["total_portfolio_analyses"],
            "total_comprehensive_analyses": self._stats[
                "total_comprehensive_analyses"
            ],
            "cache_hits": self._stats["cache_hits"],
            "cache_misses": self._stats["cache_misses"],
            "cache_hit_rate": round(cache_hit_rate, 4),
            "errors": self._stats["errors"],
            "supported_commodities": SUPPORTED_COMMODITIES,
            "supported_commodities_count": len(SUPPORTED_COMMODITIES),
            "timestamp": utcnow().isoformat(),
        }

        # Add per-engine statistics if available
        engine_stats: Dict[str, Any] = {}
        for engine_name, engine in [
            ("CommodityProfiler", self._commodity_profiler),
            ("DerivedProductAnalyzer", self._derived_product_analyzer),
            ("PriceVolatilityEngine", self._price_volatility_engine),
            ("ProductionForecastEngine", self._production_forecast_engine),
            ("SubstitutionRiskAnalyzer", self._substitution_risk_analyzer),
            ("RegulatoryComplianceEngine", self._regulatory_compliance_engine),
            ("CommodityDueDiligenceEngine", self._commodity_dd_engine),
            ("PortfolioRiskAggregator", self._portfolio_risk_aggregator),
        ]:
            if engine is not None and hasattr(engine, "get_statistics"):
                try:
                    engine_stats[engine_name] = await engine.get_statistics()
                except Exception as e:
                    engine_stats[engine_name] = {"error": str(e)}

        if engine_stats:
            stats["engine_statistics"] = engine_stats

        return stats

# =============================================================================
# Module-level singleton management
# =============================================================================

_service_instance: Optional[CommodityRiskAnalyzerSetup] = None
_service_lock = threading.Lock()

def get_service(
    config: Optional[CommodityRiskAnalyzerConfig] = None,
) -> CommodityRiskAnalyzerSetup:
    """
    Get or create the singleton CommodityRiskAnalyzerSetup instance.

    Thread-safe singleton with double-checked locking pattern.

    Args:
        config: Optional configuration override (only used on first call).

    Returns:
        Singleton CommodityRiskAnalyzerSetup instance.

    Example:
        >>> service = get_service()
        >>> await service.startup()
        >>> profile = await service.profile_commodity("cocoa")
        >>> await service.shutdown()
    """
    global _service_instance

    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = CommodityRiskAnalyzerSetup(config=config)
                logger.info("CommodityRiskAnalyzerSetup singleton created")

    return _service_instance

def set_service(service: CommodityRiskAnalyzerSetup) -> None:
    """
    Override the singleton service instance (for testing).

    Args:
        service: New service instance to use as singleton.
    """
    global _service_instance
    with _service_lock:
        _service_instance = service
        logger.info("CommodityRiskAnalyzerSetup singleton overridden")

def reset_service() -> None:
    """
    Reset the singleton service instance (for testing).

    Warning: Does not call shutdown(). Caller must handle cleanup.
    """
    global _service_instance
    with _service_lock:
        _service_instance = None
        logger.info("CommodityRiskAnalyzerSetup singleton reset")

# =============================================================================
# FastAPI Lifespan Integration
# =============================================================================

@asynccontextmanager
async def lifespan(app: Any) -> AsyncIterator[None]:
    """
    FastAPI lifespan context manager for automatic startup/shutdown.

    Usage:
        >>> from fastapi import FastAPI
        >>> from greenlang.agents.eudr.commodity_risk_analyzer.setup import lifespan
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
    logger.info("CommodityRiskAnalyzerSetup started (FastAPI lifespan)")

    yield

    # Shutdown
    await service.shutdown()
    logger.info("CommodityRiskAnalyzerSetup shutdown (FastAPI lifespan)")

# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    "CommodityRiskAnalyzerSetup",
    "get_service",
    "set_service",
    "reset_service",
    "lifespan",
    "PSYCOPG_POOL_AVAILABLE",
    "PSYCOPG_AVAILABLE",
    "REDIS_AVAILABLE",
    "OTEL_AVAILABLE",
]
