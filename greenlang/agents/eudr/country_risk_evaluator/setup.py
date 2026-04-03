# -*- coding: utf-8 -*-
"""
CountryRiskEvaluatorService - Facade for AGENT-EUDR-016

Unified service facade orchestrating all 8 engines of the Country Risk
Evaluator Agent. Provides a single entry point for country risk assessment,
commodity-specific risk analysis, deforestation hotspot detection, governance
index evaluation, due diligence classification, trade flow analysis, risk
report generation, and regulatory update tracking.

Engines (8):
    1. CountryRiskScorer             - Composite country risk scoring (Feature 1)
    2. CommodityRiskAnalyzer         - Commodity-specific risk analysis (Feature 2)
    3. DeforestationHotspotDetector  - Sub-national hotspot detection (Feature 3)
    4. GovernanceIndexEngine         - Governance index evaluation (Feature 4)
    5. DueDiligenceClassifier        - 3-tier DD classification (Feature 5)
    6. TradeFlowAnalyzer             - Bilateral trade flow analysis (Feature 6)
    7. RiskReportGenerator           - Audit-ready report generation (Feature 7)
    8. RegulatoryUpdateTracker       - EC regulatory update tracking (Feature 8)

Reference Data (4):
    - country_risk_database: 60+ countries with EC benchmarking classifications
    - governance_indices: WGI, CPI, forest governance scores
    - trade_flow_data: Major bilateral trade flows, transshipment hubs
    - (additional reference data in reference_data/ package)

Singleton Pattern:
    Thread-safe singleton with double-checked locking via ``get_service()``.

FastAPI Integration:
    Use the ``lifespan`` async context manager with
    ``FastAPI(lifespan=lifespan)`` for automatic startup/shutdown.

Example:
    >>> from greenlang.agents.eudr.country_risk_evaluator.setup import (
    ...     CountryRiskEvaluatorService,
    ...     get_service,
    ... )
    >>> service = get_service()
    >>> await service.startup()
    >>> health = await service.health_check()
    >>> assert health.status == "healthy"
    >>>
    >>> # Full country assessment
    >>> assessment = await service.assess_country(
    ...     country_code="BR",
    ...     include_commodities=True,
    ...     include_hotspots=True,
    ...     include_governance=True,
    ... )
    >>> assert assessment.risk_level == "high"
    >>>
    >>> # Commodity-specific risk
    >>> commodity_risk = await service.analyze_commodity_risk(
    ...     country_code="ID",
    ...     commodity="oil_palm",
    ... )
    >>>
    >>> # Compare countries
    >>> comparison = await service.compare_countries(
    ...     country_codes=["BR", "ID", "MY"],
    ...     metric="risk_score",
    ... )
    >>>
    >>> await service.shutdown()

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-016
Agent ID: GL-EUDR-CRE-016
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

from greenlang.agents.eudr.country_risk_evaluator.config import (
    CountryRiskEvaluatorConfig,
    get_config,
    set_config,
    reset_config,
)
from greenlang.agents.eudr.country_risk_evaluator.provenance import (
    ProvenanceTracker,
    get_provenance_tracker,
)
from greenlang.agents.eudr.country_risk_evaluator.metrics import (
    PROMETHEUS_AVAILABLE,
    record_assessment_completed,
    record_commodity_analysis,
    record_hotspot_detected,
    record_classification_completed,
    record_report_generated,
    record_trade_analysis,
    record_regulatory_update,
    record_api_error,
    observe_assessment_duration,
    observe_commodity_analysis_duration,
    observe_hotspot_detection_duration,
    observe_classification_duration,
    observe_report_generation_duration,
    set_active_hotspots,
    set_countries_assessed,
    set_high_risk_countries,
    set_pending_reclassifications,
    set_stale_assessments,
)

# ---------------------------------------------------------------------------
# Internal imports: models
# ---------------------------------------------------------------------------

from greenlang.agents.eudr.country_risk_evaluator.models import (
    VERSION,
    MAX_RISK_SCORE,
    MIN_RISK_SCORE,
    MAX_BATCH_SIZE,
    EUDR_CUTOFF_DATE,
    EC_BENCHMARK_URL,
    SUPPORTED_COMMODITIES,
    SUPPORTED_OUTPUT_FORMATS,
    SUPPORTED_REPORT_LANGUAGES,
    DEFAULT_FACTOR_WEIGHTS,
    SUPPORTED_COUNTRIES,
    RiskLevel,
    DueDiligenceLevel,
    CommodityType,
    ForestType,
    GovernanceIndicator,
    HotspotSeverity,
    DeforestationDriver,
    TradeFlowDirection,
    ReportFormat,
    ReportType,
    RegulatoryStatus,
    AssessmentConfidence,
    TrendDirection,
    CertificationScheme,
    DataSource,
    CountryRiskAssessment,
    CommodityRiskProfile,
    DeforestationHotspot,
    GovernanceIndex,
    DueDiligenceClassification,
    TradeFlow,
    RiskReport,
    RegulatoryUpdate,
    RiskFactor,
    RiskHistory,
    CertificationRecord,
    AuditLogEntry,
    AssessCountryRequest,
    AnalyzeCommodityRequest,
    DetectHotspotsRequest,
    EvaluateGovernanceRequest,
    ClassifyDueDiligenceRequest,
    AnalyzeTradeFlowRequest,
    GenerateReportRequest,
    TrackRegulatoryRequest,
    CompareCountriesRequest,
    GetTrendsRequest,
    CostEstimateRequest,
    MatrixRequest,
    ClusteringRequest,
    ImpactAssessmentRequest,
    SearchRequest,
    CountryRiskResponse,
    CommodityRiskResponse,
    HotspotResponse,
    GovernanceResponse,
    DueDiligenceResponse,
    TradeFlowResponse,
    ReportResponse,
    RegulatoryResponse,
    ComparisonResponse,
    TrendResponse,
    CostEstimateResponse,
    MatrixResponse,
    ClusteringResponse,
    ImpactResponse,
    HealthResponse,
)

# ---------------------------------------------------------------------------
# Internal imports: reference data
# ---------------------------------------------------------------------------

from greenlang.agents.eudr.country_risk_evaluator.reference_data import (
    COUNTRY_RISK_DATABASE,
    DATA_VERSION,
    EUDR_COMMODITIES,
    get_country_risk_data,
    get_high_risk_countries,
    get_low_risk_countries,
    get_standard_risk_countries,
    WORLD_BANK_WGI,
    TI_CPI_SCORES,
    FOREST_GOVERNANCE_SCORES,
    ENFORCEMENT_EFFECTIVENESS,
    get_wgi_score,
    get_cpi_score,
    get_forest_governance,
    get_enforcement_score,
    MAJOR_TRADE_FLOWS,
    TRANSSHIPMENT_HUBS,
    HS_CODE_MAPPING,
    COMMODITY_PRODUCTION_DATA,
    CERTIFICATION_COVERAGE,
    get_trade_flows,
    get_transshipment_risk,
    map_hs_to_commodity,
    get_production_volume,
    get_certification_coverage,
)

# ---------------------------------------------------------------------------
# Engine imports (graceful fallback)
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.country_risk_evaluator.country_risk_scorer import (
        CountryRiskScorer,
    )

    _COUNTRY_RISK_SCORER_AVAILABLE = True
except ImportError:
    CountryRiskScorer = None  # type: ignore[assignment,misc]
    _COUNTRY_RISK_SCORER_AVAILABLE = False

try:
    from greenlang.agents.eudr.country_risk_evaluator.commodity_risk_analyzer import (
        CommodityRiskAnalyzer,
    )

    _COMMODITY_RISK_ANALYZER_AVAILABLE = True
except ImportError:
    CommodityRiskAnalyzer = None  # type: ignore[assignment,misc]
    _COMMODITY_RISK_ANALYZER_AVAILABLE = False

try:
    from greenlang.agents.eudr.country_risk_evaluator.deforestation_hotspot_detector import (
        DeforestationHotspotDetector,
    )

    _DEFORESTATION_HOTSPOT_DETECTOR_AVAILABLE = True
except ImportError:
    DeforestationHotspotDetector = None  # type: ignore[assignment,misc]
    _DEFORESTATION_HOTSPOT_DETECTOR_AVAILABLE = False

try:
    from greenlang.agents.eudr.country_risk_evaluator.governance_index_engine import (
        GovernanceIndexEngine,
    )

    _GOVERNANCE_INDEX_ENGINE_AVAILABLE = True
except ImportError:
    GovernanceIndexEngine = None  # type: ignore[assignment,misc]
    _GOVERNANCE_INDEX_ENGINE_AVAILABLE = False

try:
    from greenlang.agents.eudr.country_risk_evaluator.due_diligence_classifier import (
        DueDiligenceClassifier,
    )

    _DUE_DILIGENCE_CLASSIFIER_AVAILABLE = True
except ImportError:
    DueDiligenceClassifier = None  # type: ignore[assignment,misc]
    _DUE_DILIGENCE_CLASSIFIER_AVAILABLE = False

try:
    from greenlang.agents.eudr.country_risk_evaluator.trade_flow_analyzer import (
        TradeFlowAnalyzer,
    )

    _TRADE_FLOW_ANALYZER_AVAILABLE = True
except ImportError:
    TradeFlowAnalyzer = None  # type: ignore[assignment,misc]
    _TRADE_FLOW_ANALYZER_AVAILABLE = False

try:
    from greenlang.agents.eudr.country_risk_evaluator.risk_report_generator import (
        RiskReportGenerator,
    )

    _RISK_REPORT_GENERATOR_AVAILABLE = True
except ImportError:
    RiskReportGenerator = None  # type: ignore[assignment,misc]
    _RISK_REPORT_GENERATOR_AVAILABLE = False

try:
    from greenlang.agents.eudr.country_risk_evaluator.regulatory_update_tracker import (
        RegulatoryUpdateTracker,
    )

    _REGULATORY_UPDATE_TRACKER_AVAILABLE = True
except ImportError:
    RegulatoryUpdateTracker = None  # type: ignore[assignment,misc]
    _REGULATORY_UPDATE_TRACKER_AVAILABLE = False

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"
_AGENT_ID = "GL-EUDR-CRE-016"
_ENGINE_COUNT = 8

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _calculate_sha256(data: Any) -> str:
    """Calculate SHA-256 hash of JSON-serialized data."""
    if isinstance(data, str):
        payload = data
    elif isinstance(data, bytes):
        payload = data.decode("utf-8", errors="replace")
    else:
        payload = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def _safe_decimal(value: Any, default: Decimal = Decimal("0.0")) -> Decimal:
    """Safely convert value to Decimal."""
    try:
        return Decimal(str(value))
    except Exception:
        return default

# =============================================================================
# FACADE: CountryRiskEvaluatorService
# =============================================================================

class CountryRiskEvaluatorService:
    """
    CountryRiskEvaluatorService orchestrates all 8 engines of AGENT-EUDR-016.

    This facade provides a single, thread-safe entry point for all country
    risk evaluation operations per EUDR Articles 10, 11, 13, 29, and 31.

    Architecture:
        - Lazy initialization of all 8 engines (on first use)
        - Thread-safe singleton pattern with double-checked locking
        - PostgreSQL connection pooling via psycopg_pool
        - Redis caching for frequently accessed reference data
        - OpenTelemetry distributed tracing integration
        - Prometheus metrics for all operations
        - SHA-256 provenance hashing for audit trails

    Engines:
        1. CountryRiskScorer: Composite country risk scoring (6 weighted factors)
        2. CommodityRiskAnalyzer: Commodity-specific risk analysis (7 EUDR commodities)
        3. DeforestationHotspotDetector: Sub-national hotspot detection (DBSCAN clustering)
        4. GovernanceIndexEngine: Governance index evaluation (WGI, CPI, forest governance)
        5. DueDiligenceClassifier: 3-tier DD classification (simplified/standard/enhanced)
        6. TradeFlowAnalyzer: Bilateral trade flow analysis (re-export risk detection)
        7. RiskReportGenerator: Audit-ready report generation (PDF/JSON/HTML)
        8. RegulatoryUpdateTracker: EC regulatory update tracking (benchmarking list monitoring)

    Attributes:
        config: Current configuration instance
        db_pool: Async PostgreSQL connection pool (psycopg_pool)
        redis_client: Async Redis client (redis.asyncio)
        provenance_tracker: SHA-256 provenance tracking
        tracer: OpenTelemetry tracer (optional)

    Example:
        >>> service = get_service()
        >>> await service.startup()
        >>>
        >>> # Full country assessment
        >>> assessment = await service.assess_country(
        ...     country_code="BR",
        ...     include_commodities=True,
        ...     include_hotspots=True,
        ...     include_governance=True,
        ... )
        >>>
        >>> # Commodity-specific risk
        >>> commodity_risk = await service.analyze_commodity_risk(
        ...     country_code="ID",
        ...     commodity="oil_palm",
        ... )
        >>>
        >>> # Generate risk report
        >>> report = await service.generate_report(
        ...     country_codes=["BR", "ID", "MY"],
        ...     report_type="comparison",
        ...     output_format="pdf",
        ... )
        >>>
        >>> await service.shutdown()
    """

    def __init__(
        self,
        config: Optional[CountryRiskEvaluatorConfig] = None,
        *,
        db_pool: Optional[AsyncConnectionPool] = None,
        redis_client: Optional[aioredis.Redis] = None,  # type: ignore[name-defined]
    ) -> None:
        """
        Initialize CountryRiskEvaluatorService.

        Args:
            config: Optional configuration override. Defaults to global config.
            db_pool: Optional pre-initialized PostgreSQL connection pool.
            redis_client: Optional pre-initialized Redis client.
        """
        self._config = config or get_config()
        self._db_pool = db_pool
        self._redis_client = redis_client
        self._provenance_tracker = get_provenance_tracker()

        # OpenTelemetry tracer (optional)
        if OTEL_AVAILABLE and otel_trace:
            self._tracer = otel_trace.get_tracer(__name__, version=_MODULE_VERSION)
        else:
            self._tracer = None

        # Engine instances (lazy initialized)
        self._country_risk_scorer: Optional[CountryRiskScorer] = None
        self._commodity_risk_analyzer: Optional[CommodityRiskAnalyzer] = None
        self._deforestation_hotspot_detector: Optional[
            DeforestationHotspotDetector
        ] = None
        self._governance_index_engine: Optional[GovernanceIndexEngine] = None
        self._due_diligence_classifier: Optional[DueDiligenceClassifier] = None
        self._trade_flow_analyzer: Optional[TradeFlowAnalyzer] = None
        self._risk_report_generator: Optional[RiskReportGenerator] = None
        self._regulatory_update_tracker: Optional[RegulatoryUpdateTracker] = None

        # Lifecycle state
        self._started = False
        self._startup_lock = asyncio.Lock()
        self._shutdown_lock = asyncio.Lock()

        logger.info(
            f"CountryRiskEvaluatorService initialized (version={_MODULE_VERSION}, agent_id={_AGENT_ID})"
        )

    # -----------------------------------------------------------------------
    # Lifecycle management
    # -----------------------------------------------------------------------

    async def startup(self) -> None:
        """
        Initialize all resources (database, Redis, engines).

        This method is idempotent and thread-safe. Multiple calls are safe.
        """
        async with self._startup_lock:
            if self._started:
                logger.debug("CountryRiskEvaluatorService already started, skipping startup")
                return

            logger.info("Starting CountryRiskEvaluatorService...")
            start_time = time.monotonic()

            try:
                # 1. Initialize database connection pool
                if self._db_pool is None and PSYCOPG_POOL_AVAILABLE:
                    logger.debug("Creating PostgreSQL connection pool...")
                    self._db_pool = AsyncConnectionPool(
                        conninfo=self._config.database_url,
                        min_size=self._config.pool_min_size,
                        max_size=self._config.pool_size,
                        timeout=30.0,
                        max_idle=300.0,
                        max_lifetime=3600.0,
                    )
                    await self._db_pool.open()
                    logger.info(
                        f"PostgreSQL pool opened (min={self._config.pool_min_size}, max={self._config.pool_size})"
                    )

                # 2. Initialize Redis client
                if self._redis_client is None and REDIS_AVAILABLE and aioredis:
                    logger.debug("Creating Redis client...")
                    self._redis_client = await aioredis.from_url(
                        self._config.redis_url,
                        decode_responses=True,
                        max_connections=self._config.redis_max_connections,
                        socket_timeout=5.0,
                        socket_connect_timeout=5.0,
                    )
                    # Ping to verify connection
                    await self._redis_client.ping()
                    logger.info("Redis client connected")

                # 3. Lazy engine initialization will happen on first use
                logger.debug("Engines will be initialized on first use (lazy loading)")

                # 4. Mark as started
                self._started = True
                duration_ms = (time.monotonic() - start_time) * 1000

                logger.info(
                    f"CountryRiskEvaluatorService started successfully (duration={duration_ms:.2f}ms)"
                )

            except Exception as e:
                logger.error("Startup failed: %s", e, exc_info=True)
                record_api_error("startup", str(e))
                raise

    async def shutdown(self) -> None:
        """
        Gracefully shutdown all resources.

        Closes database pool, Redis client, and cleans up engine resources.
        """
        async with self._shutdown_lock:
            if not self._started:
                logger.debug("CountryRiskEvaluatorService not started, skipping shutdown")
                return

            logger.info("Shutting down CountryRiskEvaluatorService...")

            try:
                # 1. Shutdown engines
                for engine_name, engine in [
                    ("CountryRiskScorer", self._country_risk_scorer),
                    ("CommodityRiskAnalyzer", self._commodity_risk_analyzer),
                    ("DeforestationHotspotDetector", self._deforestation_hotspot_detector),
                    ("GovernanceIndexEngine", self._governance_index_engine),
                    ("DueDiligenceClassifier", self._due_diligence_classifier),
                    ("TradeFlowAnalyzer", self._trade_flow_analyzer),
                    ("RiskReportGenerator", self._risk_report_generator),
                    ("RegulatoryUpdateTracker", self._regulatory_update_tracker),
                ]:
                    if engine is not None:
                        try:
                            if hasattr(engine, "shutdown"):
                                await engine.shutdown()
                            logger.debug("%s shutdown complete", engine_name)
                        except Exception as e:
                            logger.warning("Error shutting down %s: %s", engine_name, e)

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
                        logger.warning("Error closing PostgreSQL pool: %s", e)

                # 4. Mark as shutdown
                self._started = False

                logger.info("CountryRiskEvaluatorService shutdown complete")

            except Exception as e:
                logger.error("Shutdown error: %s", e, exc_info=True)

    # -----------------------------------------------------------------------
    # Engine lazy initialization
    # -----------------------------------------------------------------------

    async def _ensure_country_risk_scorer(self) -> CountryRiskScorer:
        """Lazy initialize CountryRiskScorer engine."""
        if self._country_risk_scorer is None:
            if not _COUNTRY_RISK_SCORER_AVAILABLE or CountryRiskScorer is None:
                raise RuntimeError("CountryRiskScorer engine not available")
            logger.debug("Initializing CountryRiskScorer engine...")
            self._country_risk_scorer = CountryRiskScorer(
                config=self._config,
                db_pool=self._db_pool,
                redis_client=self._redis_client,
            )
            await self._country_risk_scorer.startup()
            logger.info("CountryRiskScorer engine initialized")
        return self._country_risk_scorer

    async def _ensure_commodity_risk_analyzer(self) -> CommodityRiskAnalyzer:
        """Lazy initialize CommodityRiskAnalyzer engine."""
        if self._commodity_risk_analyzer is None:
            if not _COMMODITY_RISK_ANALYZER_AVAILABLE or CommodityRiskAnalyzer is None:
                raise RuntimeError("CommodityRiskAnalyzer engine not available")
            logger.debug("Initializing CommodityRiskAnalyzer engine...")
            self._commodity_risk_analyzer = CommodityRiskAnalyzer(
                config=self._config,
                db_pool=self._db_pool,
                redis_client=self._redis_client,
            )
            await self._commodity_risk_analyzer.startup()
            logger.info("CommodityRiskAnalyzer engine initialized")
        return self._commodity_risk_analyzer

    async def _ensure_deforestation_hotspot_detector(
        self,
    ) -> DeforestationHotspotDetector:
        """Lazy initialize DeforestationHotspotDetector engine."""
        if self._deforestation_hotspot_detector is None:
            if (
                not _DEFORESTATION_HOTSPOT_DETECTOR_AVAILABLE
                or DeforestationHotspotDetector is None
            ):
                raise RuntimeError("DeforestationHotspotDetector engine not available")
            logger.debug("Initializing DeforestationHotspotDetector engine...")
            self._deforestation_hotspot_detector = DeforestationHotspotDetector(
                config=self._config,
                db_pool=self._db_pool,
                redis_client=self._redis_client,
            )
            await self._deforestation_hotspot_detector.startup()
            logger.info("DeforestationHotspotDetector engine initialized")
        return self._deforestation_hotspot_detector

    async def _ensure_governance_index_engine(self) -> GovernanceIndexEngine:
        """Lazy initialize GovernanceIndexEngine."""
        if self._governance_index_engine is None:
            if not _GOVERNANCE_INDEX_ENGINE_AVAILABLE or GovernanceIndexEngine is None:
                raise RuntimeError("GovernanceIndexEngine not available")
            logger.debug("Initializing GovernanceIndexEngine...")
            self._governance_index_engine = GovernanceIndexEngine(
                config=self._config,
                db_pool=self._db_pool,
                redis_client=self._redis_client,
            )
            await self._governance_index_engine.startup()
            logger.info("GovernanceIndexEngine initialized")
        return self._governance_index_engine

    async def _ensure_due_diligence_classifier(self) -> DueDiligenceClassifier:
        """Lazy initialize DueDiligenceClassifier engine."""
        if self._due_diligence_classifier is None:
            if (
                not _DUE_DILIGENCE_CLASSIFIER_AVAILABLE
                or DueDiligenceClassifier is None
            ):
                raise RuntimeError("DueDiligenceClassifier engine not available")
            logger.debug("Initializing DueDiligenceClassifier engine...")
            self._due_diligence_classifier = DueDiligenceClassifier(
                config=self._config,
                db_pool=self._db_pool,
                redis_client=self._redis_client,
            )
            await self._due_diligence_classifier.startup()
            logger.info("DueDiligenceClassifier engine initialized")
        return self._due_diligence_classifier

    async def _ensure_trade_flow_analyzer(self) -> TradeFlowAnalyzer:
        """Lazy initialize TradeFlowAnalyzer engine."""
        if self._trade_flow_analyzer is None:
            if not _TRADE_FLOW_ANALYZER_AVAILABLE or TradeFlowAnalyzer is None:
                raise RuntimeError("TradeFlowAnalyzer engine not available")
            logger.debug("Initializing TradeFlowAnalyzer engine...")
            self._trade_flow_analyzer = TradeFlowAnalyzer(
                config=self._config,
                db_pool=self._db_pool,
                redis_client=self._redis_client,
            )
            await self._trade_flow_analyzer.startup()
            logger.info("TradeFlowAnalyzer engine initialized")
        return self._trade_flow_analyzer

    async def _ensure_risk_report_generator(self) -> RiskReportGenerator:
        """Lazy initialize RiskReportGenerator engine."""
        if self._risk_report_generator is None:
            if not _RISK_REPORT_GENERATOR_AVAILABLE or RiskReportGenerator is None:
                raise RuntimeError("RiskReportGenerator engine not available")
            logger.debug("Initializing RiskReportGenerator engine...")
            self._risk_report_generator = RiskReportGenerator(
                config=self._config,
                db_pool=self._db_pool,
                redis_client=self._redis_client,
            )
            await self._risk_report_generator.startup()
            logger.info("RiskReportGenerator engine initialized")
        return self._risk_report_generator

    async def _ensure_regulatory_update_tracker(self) -> RegulatoryUpdateTracker:
        """Lazy initialize RegulatoryUpdateTracker engine."""
        if self._regulatory_update_tracker is None:
            if (
                not _REGULATORY_UPDATE_TRACKER_AVAILABLE
                or RegulatoryUpdateTracker is None
            ):
                raise RuntimeError("RegulatoryUpdateTracker engine not available")
            logger.debug("Initializing RegulatoryUpdateTracker engine...")
            self._regulatory_update_tracker = RegulatoryUpdateTracker(
                config=self._config,
                db_pool=self._db_pool,
                redis_client=self._redis_client,
            )
            await self._regulatory_update_tracker.startup()
            logger.info("RegulatoryUpdateTracker engine initialized")
        return self._regulatory_update_tracker

    # -----------------------------------------------------------------------
    # Public API: Country Risk Assessment
    # -----------------------------------------------------------------------

    async def assess_country(
        self,
        country_code: str,
        *,
        assessment_date: Optional[datetime] = None,
        custom_weights: Optional[Dict[str, float]] = None,
        include_commodities: bool = False,
        include_hotspots: bool = False,
        include_governance: bool = True,
        confidence_threshold: Optional[float] = None,
    ) -> CountryRiskResponse:
        """
        Perform comprehensive country risk assessment.

        Orchestrates CountryRiskScorer, GovernanceIndexEngine, and optionally
        CommodityRiskAnalyzer and DeforestationHotspotDetector for a complete
        country risk profile.

        Args:
            country_code: ISO 3166-1 alpha-2 country code (e.g., "BR", "ID")
            assessment_date: Optional assessment date (defaults to today)
            custom_weights: Optional custom factor weights (must sum to 1.0)
            include_commodities: Include commodity-specific risk analysis
            include_hotspots: Include deforestation hotspot detection
            include_governance: Include governance index evaluation
            confidence_threshold: Optional confidence threshold override

        Returns:
            CountryRiskResponse with complete risk assessment

        Raises:
            ValueError: If country_code is invalid or weights don't sum to 1.0
            RuntimeError: If required engines are not available
        """
        start_time = time.monotonic()
        logger.info("Starting country risk assessment for %s", country_code)

        try:
            # 1. Validate country code
            if country_code not in SUPPORTED_COUNTRIES:
                raise ValueError(f"Unsupported country code: {country_code}")

            assessment_date = assessment_date or utcnow()

            # 2. Get base country risk score (CountryRiskScorer)
            scorer = await self._ensure_country_risk_scorer()
            base_assessment = await scorer.assess_country(
                country_code=country_code,
                assessment_date=assessment_date,
                custom_weights=custom_weights,
                confidence_threshold=confidence_threshold,
            )

            # 3. Get governance index (optional)
            governance_index = None
            if include_governance:
                gov_engine = await self._ensure_governance_index_engine()
                governance_index = await gov_engine.evaluate_governance(
                    country_code=country_code, assessment_date=assessment_date
                )

            # 4. Get commodity risk profiles (optional)
            commodity_profiles = []
            if include_commodities:
                commodity_analyzer = await self._ensure_commodity_risk_analyzer()
                country_data = get_country_risk_data(country_code)
                if country_data and "commodity_production" in country_data:
                    for commodity in country_data["commodity_production"]:
                        commodity_profile = await commodity_analyzer.analyze_commodity(
                            country_code=country_code,
                            commodity=commodity,
                            assessment_date=assessment_date,
                        )
                        commodity_profiles.append(commodity_profile)

            # 5. Detect deforestation hotspots (optional)
            hotspots = []
            if include_hotspots:
                hotspot_detector = await self._ensure_deforestation_hotspot_detector()
                hotspots_response = await hotspot_detector.detect_hotspots(
                    country_code=country_code,
                    start_date=assessment_date.replace(year=assessment_date.year - 1),
                    end_date=assessment_date,
                )
                hotspots = hotspots_response.hotspots if hotspots_response else []

            # 6. Classify due diligence level
            dd_classifier = await self._ensure_due_diligence_classifier()
            dd_classification = await dd_classifier.classify_due_diligence(
                country_code=country_code, risk_score=base_assessment.risk_score
            )

            # 7. Create provenance record
            provenance_hash = self._provenance_tracker.record_action(
                entity_type="country_risk_assessment",
                entity_id=country_code,
                action="assess_country",
                metadata={
                    "country_code": country_code,
                    "risk_score": float(base_assessment.risk_score),
                    "risk_level": base_assessment.risk_level.value,
                    "dd_level": dd_classification.due_diligence_level.value,
                    "include_commodities": include_commodities,
                    "include_hotspots": include_hotspots,
                    "include_governance": include_governance,
                },
            )

            # 8. Record metrics
            duration_sec = time.monotonic() - start_time
            observe_assessment_duration(duration_sec)
            record_assessment_completed(
                country_code=country_code,
                risk_level=base_assessment.risk_level.value,
                status="success",
            )

            logger.info(
                f"Country risk assessment complete for {country_code}: "
                f"risk_level={base_assessment.risk_level.value}, "
                f"risk_score={base_assessment.risk_score}, "
                f"duration={duration_sec:.3f}s"
            )

            return CountryRiskResponse(
                success=True,
                assessment=base_assessment,
                governance_index=governance_index,
                commodity_profiles=commodity_profiles,
                hotspots=hotspots,
                dd_classification=dd_classification,
                provenance_hash=provenance_hash,
                processing_time_ms=duration_sec * 1000,
            )

        except Exception as e:
            logger.error("Country risk assessment failed for %s: %s", country_code, e, exc_info=True)
            record_api_error("assess_country", str(e))
            raise

    async def analyze_commodity_risk(
        self,
        country_code: str,
        commodity: str,
        *,
        assessment_date: Optional[datetime] = None,
        include_seasonal: bool = False,
        include_certification: bool = True,
    ) -> CommodityRiskResponse:
        """
        Analyze commodity-specific risk for a country.

        Delegates to CommodityRiskAnalyzer engine for detailed commodity risk
        profiling including production volume, deforestation correlation,
        certification effectiveness, and supply chain complexity.

        Args:
            country_code: ISO 3166-1 alpha-2 country code
            commodity: EUDR commodity (cattle, cocoa, coffee, oil_palm, rubber, soya, wood)
            assessment_date: Optional assessment date
            include_seasonal: Include seasonal risk variation analysis
            include_certification: Include certification scheme effectiveness

        Returns:
            CommodityRiskResponse with detailed commodity risk profile

        Raises:
            ValueError: If country_code or commodity is invalid
        """
        start_time = time.monotonic()
        logger.info("Analyzing commodity risk: %s / %s", country_code, commodity)

        try:
            # Validate inputs
            if country_code not in SUPPORTED_COUNTRIES:
                raise ValueError(f"Unsupported country code: {country_code}")
            if commodity not in SUPPORTED_COMMODITIES:
                raise ValueError(f"Unsupported commodity: {commodity}")

            assessment_date = assessment_date or utcnow()

            # Delegate to CommodityRiskAnalyzer
            commodity_analyzer = await self._ensure_commodity_risk_analyzer()
            commodity_profile = await commodity_analyzer.analyze_commodity(
                country_code=country_code,
                commodity=commodity,
                assessment_date=assessment_date,
                include_seasonal=include_seasonal,
                include_certification=include_certification,
            )

            # Record metrics
            duration_sec = time.monotonic() - start_time
            observe_commodity_analysis_duration(duration_sec)
            record_commodity_analysis(
                country_code=country_code, commodity=commodity, status="success"
            )

            logger.info(
                f"Commodity risk analysis complete: {country_code}/{commodity}, "
                f"duration={duration_sec:.3f}s"
            )

            return CommodityRiskResponse(
                success=True,
                commodity_profile=commodity_profile,
                processing_time_ms=duration_sec * 1000,
            )

        except Exception as e:
            logger.error("Commodity risk analysis failed: %s", e, exc_info=True)
            record_api_error("analyze_commodity_risk", str(e))
            raise

    async def detect_hotspots(
        self,
        country_code: str,
        *,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        clustering_enabled: bool = True,
        fire_correlation_enabled: bool = True,
        protected_area_check: bool = True,
    ) -> HotspotResponse:
        """
        Detect deforestation hotspots within a country.

        Delegates to DeforestationHotspotDetector for DBSCAN clustering,
        fire alert correlation, and protected area proximity analysis.

        Args:
            country_code: ISO 3166-1 alpha-2 country code
            start_date: Start date for hotspot detection (defaults to 1 year ago)
            end_date: End date for hotspot detection (defaults to today)
            clustering_enabled: Enable DBSCAN spatial clustering
            fire_correlation_enabled: Correlate with FIRMS/VIIRS fire alerts
            protected_area_check: Check proximity to protected areas

        Returns:
            HotspotResponse with detected hotspots

        Raises:
            ValueError: If country_code is invalid
        """
        start_time = time.monotonic()
        logger.info("Detecting deforestation hotspots for %s", country_code)

        try:
            if country_code not in SUPPORTED_COUNTRIES:
                raise ValueError(f"Unsupported country code: {country_code}")

            end_date = end_date or utcnow()
            start_date = start_date or end_date.replace(year=end_date.year - 1)

            # Delegate to DeforestationHotspotDetector
            hotspot_detector = await self._ensure_deforestation_hotspot_detector()
            hotspots_response = await hotspot_detector.detect_hotspots(
                country_code=country_code,
                start_date=start_date,
                end_date=end_date,
                clustering_enabled=clustering_enabled,
                fire_correlation_enabled=fire_correlation_enabled,
                protected_area_check=protected_area_check,
            )

            # Record metrics
            duration_sec = time.monotonic() - start_time
            observe_hotspot_detection_duration(duration_sec)
            record_hotspot_detected(
                country_code=country_code,
                count=len(hotspots_response.hotspots),
                status="success",
            )

            logger.info(
                f"Hotspot detection complete for {country_code}: "
                f"found {len(hotspots_response.hotspots)} hotspots, "
                f"duration={duration_sec:.3f}s"
            )

            return hotspots_response

        except Exception as e:
            logger.error("Hotspot detection failed: %s", e, exc_info=True)
            record_api_error("detect_hotspots", str(e))
            raise

    async def evaluate_governance(
        self,
        country_code: str,
        *,
        assessment_date: Optional[datetime] = None,
        include_wgi: bool = True,
        include_cpi: bool = True,
        include_forest_governance: bool = True,
        include_enforcement: bool = True,
    ) -> GovernanceResponse:
        """
        Evaluate governance indices for a country.

        Delegates to GovernanceIndexEngine for WGI, CPI, forest governance,
        and enforcement effectiveness evaluation.

        Args:
            country_code: ISO 3166-1 alpha-2 country code
            assessment_date: Optional assessment date
            include_wgi: Include World Bank WGI (6 dimensions)
            include_cpi: Include Transparency International CPI
            include_forest_governance: Include FAO/ITTO forest governance
            include_enforcement: Include enforcement effectiveness scoring

        Returns:
            GovernanceResponse with complete governance evaluation

        Raises:
            ValueError: If country_code is invalid
        """
        start_time = time.monotonic()
        logger.info("Evaluating governance for %s", country_code)

        try:
            if country_code not in SUPPORTED_COUNTRIES:
                raise ValueError(f"Unsupported country code: {country_code}")

            assessment_date = assessment_date or utcnow()

            # Delegate to GovernanceIndexEngine
            gov_engine = await self._ensure_governance_index_engine()
            governance_response = await gov_engine.evaluate_governance(
                country_code=country_code,
                assessment_date=assessment_date,
                include_wgi=include_wgi,
                include_cpi=include_cpi,
                include_forest_governance=include_forest_governance,
                include_enforcement=include_enforcement,
            )

            # Record metrics
            duration_sec = time.monotonic() - start_time

            logger.info(
                f"Governance evaluation complete for {country_code}, "
                f"duration={duration_sec:.3f}s"
            )

            return governance_response

        except Exception as e:
            logger.error("Governance evaluation failed: %s", e, exc_info=True)
            record_api_error("evaluate_governance", str(e))
            raise

    async def classify_due_diligence(
        self,
        country_code: str,
        *,
        risk_score: Optional[float] = None,
        include_cost_estimate: bool = True,
        include_audit_frequency: bool = True,
    ) -> DueDiligenceResponse:
        """
        Classify due diligence level for a country.

        Delegates to DueDiligenceClassifier for 3-tier classification
        (simplified/standard/enhanced) with cost estimation and audit
        frequency recommendation.

        Args:
            country_code: ISO 3166-1 alpha-2 country code
            risk_score: Optional risk score (will auto-calculate if not provided)
            include_cost_estimate: Include cost estimation (EUR)
            include_audit_frequency: Include audit frequency recommendation

        Returns:
            DueDiligenceResponse with classification and recommendations

        Raises:
            ValueError: If country_code is invalid
        """
        start_time = time.monotonic()
        logger.info("Classifying due diligence for %s", country_code)

        try:
            if country_code not in SUPPORTED_COUNTRIES:
                raise ValueError(f"Unsupported country code: {country_code}")

            # Auto-calculate risk score if not provided
            if risk_score is None:
                scorer = await self._ensure_country_risk_scorer()
                assessment = await scorer.assess_country(country_code=country_code)
                risk_score = float(assessment.risk_score)

            # Delegate to DueDiligenceClassifier
            dd_classifier = await self._ensure_due_diligence_classifier()
            dd_response = await dd_classifier.classify_due_diligence(
                country_code=country_code,
                risk_score=risk_score,
                include_cost_estimate=include_cost_estimate,
                include_audit_frequency=include_audit_frequency,
            )

            # Record metrics
            duration_sec = time.monotonic() - start_time
            observe_classification_duration(duration_sec)
            record_classification_completed(
                country_code=country_code,
                dd_level=dd_response.classification.due_diligence_level.value,
                status="success",
            )

            logger.info(
                f"Due diligence classification complete for {country_code}: "
                f"level={dd_response.classification.due_diligence_level.value}, "
                f"duration={duration_sec:.3f}s"
            )

            return dd_response

        except Exception as e:
            logger.error("Due diligence classification failed: %s", e, exc_info=True)
            record_api_error("classify_due_diligence", str(e))
            raise

    async def analyze_trade_flows(
        self,
        country_code: Optional[str] = None,
        *,
        commodity: Optional[str] = None,
        direction: Optional[TradeFlowDirection] = None,
        include_transshipment_risk: bool = True,
        include_concentration_risk: bool = True,
    ) -> TradeFlowResponse:
        """
        Analyze bilateral trade flows.

        Delegates to TradeFlowAnalyzer for trade flow mapping, re-export
        risk detection, and concentration risk analysis.

        Args:
            country_code: Optional country filter (origin or destination)
            commodity: Optional commodity filter
            direction: Optional direction filter (import/export/both)
            include_transshipment_risk: Include transshipment/re-export risk
            include_concentration_risk: Include HHI concentration risk

        Returns:
            TradeFlowResponse with trade flow analysis

        Raises:
            ValueError: If filters are invalid
        """
        start_time = time.monotonic()
        logger.info("Analyzing trade flows (country=%s, commodity=%s)", country_code, commodity)

        try:
            # Delegate to TradeFlowAnalyzer
            trade_analyzer = await self._ensure_trade_flow_analyzer()
            trade_response = await trade_analyzer.analyze_trade_flows(
                country_code=country_code,
                commodity=commodity,
                direction=direction,
                include_transshipment_risk=include_transshipment_risk,
                include_concentration_risk=include_concentration_risk,
            )

            # Record metrics
            duration_sec = time.monotonic() - start_time
            record_trade_analysis(status="success")

            logger.info(
                f"Trade flow analysis complete: "
                f"found {len(trade_response.trade_flows)} flows, "
                f"duration={duration_sec:.3f}s"
            )

            return trade_response

        except Exception as e:
            logger.error("Trade flow analysis failed: %s", e, exc_info=True)
            record_api_error("analyze_trade_flows", str(e))
            raise

    async def generate_report(
        self,
        country_codes: List[str],
        *,
        report_type: ReportType = ReportType.COUNTRY_PROFILE,
        output_format: ReportFormat = ReportFormat.PDF,
        language: str = "en",
        include_executive_summary: bool = True,
        include_recommendations: bool = True,
    ) -> ReportResponse:
        """
        Generate audit-ready risk report.

        Delegates to RiskReportGenerator for multi-format report generation
        (PDF/JSON/HTML) with multi-language support.

        Args:
            country_codes: List of country codes to include
            report_type: Type of report (profile/comparison/trend/matrix)
            output_format: Output format (pdf/json/html/csv/excel)
            language: Report language (en/fr/de/es/pt)
            include_executive_summary: Include executive summary section
            include_recommendations: Include recommendations section

        Returns:
            ReportResponse with generated report

        Raises:
            ValueError: If inputs are invalid
        """
        start_time = time.monotonic()
        logger.info(
            f"Generating risk report: {len(country_codes)} countries, "
            f"type={report_type.value}, format={output_format.value}"
        )

        try:
            # Delegate to RiskReportGenerator
            report_generator = await self._ensure_risk_report_generator()
            report_response = await report_generator.generate_report(
                country_codes=country_codes,
                report_type=report_type,
                output_format=output_format,
                language=language,
                include_executive_summary=include_executive_summary,
                include_recommendations=include_recommendations,
            )

            # Record metrics
            duration_sec = time.monotonic() - start_time
            observe_report_generation_duration(duration_sec)
            record_report_generated(
                report_type=report_type.value,
                output_format=output_format.value,
                status="success",
            )

            logger.info(
                f"Risk report generation complete: "
                f"report_id={report_response.report.report_id}, "
                f"duration={duration_sec:.3f}s"
            )

            return report_response

        except Exception as e:
            logger.error("Report generation failed: %s", e, exc_info=True)
            record_api_error("generate_report", str(e))
            raise

    async def track_regulatory_updates(
        self,
        *,
        since_date: Optional[datetime] = None,
        country_codes: Optional[List[str]] = None,
        regulatory_status: Optional[RegulatoryStatus] = None,
    ) -> RegulatoryResponse:
        """
        Track EC regulatory updates and country reclassifications.

        Delegates to RegulatoryUpdateTracker for EC benchmarking list
        monitoring and impact assessment.

        Args:
            since_date: Optional filter for updates since date
            country_codes: Optional country filter
            regulatory_status: Optional status filter (pending/active/superseded)

        Returns:
            RegulatoryResponse with regulatory updates

        Raises:
            ValueError: If filters are invalid
        """
        start_time = time.monotonic()
        logger.info("Tracking regulatory updates")

        try:
            # Delegate to RegulatoryUpdateTracker
            regulatory_tracker = await self._ensure_regulatory_update_tracker()
            regulatory_response = await regulatory_tracker.track_regulatory_updates(
                since_date=since_date,
                country_codes=country_codes,
                regulatory_status=regulatory_status,
            )

            # Record metrics
            duration_sec = time.monotonic() - start_time
            record_regulatory_update(count=len(regulatory_response.updates), status="success")

            logger.info(
                f"Regulatory update tracking complete: "
                f"found {len(regulatory_response.updates)} updates, "
                f"duration={duration_sec:.3f}s"
            )

            return regulatory_response

        except Exception as e:
            logger.error("Regulatory update tracking failed: %s", e, exc_info=True)
            record_api_error("track_regulatory_updates", str(e))
            raise

    # -----------------------------------------------------------------------
    # Public API: Cross-engine orchestration
    # -----------------------------------------------------------------------

    async def compare_countries(
        self,
        country_codes: List[str],
        *,
        metric: str = "risk_score",
        include_trends: bool = False,
    ) -> ComparisonResponse:
        """
        Compare multiple countries across specified metric.

        Orchestrates CountryRiskScorer and optionally other engines for
        comparative analysis.

        Args:
            country_codes: List of country codes to compare (2-20 countries)
            metric: Comparison metric (risk_score/governance/enforcement/deforestation_rate)
            include_trends: Include historical trend data

        Returns:
            ComparisonResponse with comparative analysis

        Raises:
            ValueError: If inputs are invalid
        """
        start_time = time.monotonic()
        logger.info("Comparing %s countries on metric '%s'", len(country_codes), metric)

        try:
            if not (2 <= len(country_codes) <= 20):
                raise ValueError("Must compare between 2 and 20 countries")

            # Get assessments for all countries
            scorer = await self._ensure_country_risk_scorer()
            assessments = []
            for country_code in country_codes:
                assessment = await scorer.assess_country(country_code=country_code)
                assessments.append(assessment)

            # Sort by specified metric
            if metric == "risk_score":
                assessments.sort(key=lambda a: a.risk_score, reverse=True)
            elif metric == "governance":
                assessments.sort(
                    key=lambda a: a.composite_factors.get("governance_index", 0),
                    reverse=False,
                )
            elif metric == "enforcement":
                assessments.sort(
                    key=lambda a: a.composite_factors.get("enforcement_score", 0),
                    reverse=False,
                )
            elif metric == "deforestation_rate":
                assessments.sort(
                    key=lambda a: a.composite_factors.get("deforestation_rate", 0),
                    reverse=True,
                )
            else:
                raise ValueError(f"Unsupported comparison metric: {metric}")

            duration_sec = time.monotonic() - start_time

            logger.info(
                f"Country comparison complete: {len(country_codes)} countries, "
                f"metric={metric}, duration={duration_sec:.3f}s"
            )

            return ComparisonResponse(
                success=True,
                assessments=assessments,
                comparison_metric=metric,
                processing_time_ms=duration_sec * 1000,
            )

        except Exception as e:
            logger.error("Country comparison failed: %s", e, exc_info=True)
            record_api_error("compare_countries", str(e))
            raise

    async def get_risk_matrix(
        self,
        *,
        region: Optional[str] = None,
        commodity: Optional[str] = None,
    ) -> MatrixResponse:
        """
        Generate country-commodity risk matrix.

        Creates a 2D matrix showing risk scores for multiple countries and
        commodities, useful for portfolio-level risk visualization.

        Args:
            region: Optional region filter (south_america, southeast_asia, etc.)
            commodity: Optional commodity filter

        Returns:
            MatrixResponse with risk matrix data

        Raises:
            ValueError: If filters are invalid
        """
        start_time = time.monotonic()
        logger.info("Generating risk matrix (region=%s, commodity=%s)", region, commodity)

        try:
            # Get relevant countries
            if region:
                # Filter by region (simplified - would use reference data in production)
                relevant_countries = [
                    code
                    for code in SUPPORTED_COUNTRIES
                    if get_country_risk_data(code)
                    and get_country_risk_data(code).get("region") == region
                ][:20]  # Limit to 20
            else:
                relevant_countries = get_high_risk_countries()[:10]  # Top 10 high-risk

            # Get relevant commodities
            commodities = [commodity] if commodity else SUPPORTED_COMMODITIES

            # Build matrix
            matrix_data = {}
            for country_code in relevant_countries:
                matrix_data[country_code] = {}
                for comm in commodities:
                    # Get commodity risk score
                    commodity_analyzer = await self._ensure_commodity_risk_analyzer()
                    try:
                        profile = await commodity_analyzer.analyze_commodity(
                            country_code=country_code, commodity=comm
                        )
                        matrix_data[country_code][comm] = float(profile.risk_score)
                    except Exception:
                        matrix_data[country_code][comm] = None

            duration_sec = time.monotonic() - start_time

            logger.info(
                f"Risk matrix generation complete: "
                f"{len(relevant_countries)} countries x {len(commodities)} commodities, "
                f"duration={duration_sec:.3f}s"
            )

            return MatrixResponse(
                success=True,
                matrix_data=matrix_data,
                countries=relevant_countries,
                commodities=commodities,
                processing_time_ms=duration_sec * 1000,
            )

        except Exception as e:
            logger.error("Risk matrix generation failed: %s", e, exc_info=True)
            record_api_error("get_risk_matrix", str(e))
            raise

    # -----------------------------------------------------------------------
    # Public API: Configuration and health
    # -----------------------------------------------------------------------

    async def health_check(self) -> HealthResponse:
        """
        Perform comprehensive health check of all engines.

        Returns:
            HealthResponse with overall status and individual engine statuses
        """
        start_time = time.monotonic()
        logger.debug("Performing health check")

        engine_statuses = {}
        overall_healthy = True

        # Check each engine (if initialized)
        for engine_name, engine in [
            ("CountryRiskScorer", self._country_risk_scorer),
            ("CommodityRiskAnalyzer", self._commodity_risk_analyzer),
            ("DeforestationHotspotDetector", self._deforestation_hotspot_detector),
            ("GovernanceIndexEngine", self._governance_index_engine),
            ("DueDiligenceClassifier", self._due_diligence_classifier),
            ("TradeFlowAnalyzer", self._trade_flow_analyzer),
            ("RiskReportGenerator", self._risk_report_generator),
            ("RegulatoryUpdateTracker", self._regulatory_update_tracker),
        ]:
            if engine is not None:
                try:
                    if hasattr(engine, "health_check"):
                        engine_health = await engine.health_check()
                        engine_statuses[engine_name] = engine_health.status
                        if engine_health.status != "healthy":
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
                logger.warning("Database health check failed: %s", e)
                overall_healthy = False

        # Check Redis connection
        redis_healthy = False
        if self._redis_client is not None:
            try:
                await self._redis_client.ping()
                redis_healthy = True
            except Exception as e:
                logger.warning("Redis health check failed: %s", e)
                # Redis is optional, don't mark overall as unhealthy

        duration_ms = (time.monotonic() - start_time) * 1000

        return HealthResponse(
            status="healthy" if overall_healthy else "unhealthy",
            version=_MODULE_VERSION,
            agent_id=_AGENT_ID,
            started=self._started,
            database_connected=db_healthy,
            redis_connected=redis_healthy,
            engine_statuses=engine_statuses,
            processing_time_ms=duration_ms,
        )

    def get_config(self) -> CountryRiskEvaluatorConfig:
        """Get current configuration."""
        return self._config

    async def update_config(self, config: CountryRiskEvaluatorConfig) -> None:
        """
        Update configuration (hot-reload).

        Note: Some settings may require service restart to take effect.

        Args:
            config: New configuration instance
        """
        logger.info("Updating configuration (hot-reload)")
        self._config = config
        set_config(config)

    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get aggregate statistics across all engines.

        Returns:
            Dictionary with statistics (countries_assessed, high_risk_count, etc.)
        """
        logger.debug("Gathering statistics")

        stats = {
            "version": _MODULE_VERSION,
            "agent_id": _AGENT_ID,
            "data_version": DATA_VERSION,
            "supported_countries_count": len(SUPPORTED_COUNTRIES),
            "supported_commodities_count": len(SUPPORTED_COMMODITIES),
            "high_risk_countries_count": len(get_high_risk_countries()),
            "low_risk_countries_count": len(get_low_risk_countries()),
            "standard_risk_countries_count": len(get_standard_risk_countries()),
        }

        # Add engine-specific statistics (if engines are initialized)
        if self._country_risk_scorer:
            try:
                stats["scorer_stats"] = await self._country_risk_scorer.get_statistics()
            except Exception as e:
                logger.warning("Failed to get scorer statistics: %s", e)

        # Add more engine stats as needed...

        return stats

# =============================================================================
# Module-level singleton management
# =============================================================================

_service_instance: Optional[CountryRiskEvaluatorService] = None
_service_lock = threading.Lock()

def get_service(
    config: Optional[CountryRiskEvaluatorConfig] = None,
) -> CountryRiskEvaluatorService:
    """
    Get or create the singleton CountryRiskEvaluatorService instance.

    Thread-safe singleton with double-checked locking pattern.

    Args:
        config: Optional configuration override (only used on first call)

    Returns:
        Singleton CountryRiskEvaluatorService instance

    Example:
        >>> service = get_service()
        >>> await service.startup()
        >>> assessment = await service.assess_country("BR")
        >>> await service.shutdown()
    """
    global _service_instance

    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = CountryRiskEvaluatorService(config=config)
                logger.info("CountryRiskEvaluatorService singleton created")

    return _service_instance

def set_service(service: CountryRiskEvaluatorService) -> None:
    """
    Override the singleton service instance (for testing).

    Args:
        service: New service instance to use as singleton
    """
    global _service_instance
    with _service_lock:
        _service_instance = service
        logger.info("CountryRiskEvaluatorService singleton overridden")

def reset_service() -> None:
    """
    Reset the singleton service instance (for testing).

    Warning: Does not call shutdown(). Caller must handle cleanup.
    """
    global _service_instance
    with _service_lock:
        _service_instance = None
        logger.info("CountryRiskEvaluatorService singleton reset")

# =============================================================================
# FastAPI Lifespan Integration
# =============================================================================

@asynccontextmanager
async def lifespan(app: Any) -> AsyncIterator[None]:
    """
    FastAPI lifespan context manager for automatic startup/shutdown.

    Usage:
        >>> from fastapi import FastAPI
        >>> from greenlang.agents.eudr.country_risk_evaluator.setup import lifespan
        >>>
        >>> app = FastAPI(lifespan=lifespan)

    Args:
        app: FastAPI application instance

    Yields:
        None (context manager)
    """
    # Startup
    service = get_service()
    await service.startup()
    logger.info("CountryRiskEvaluatorService started (FastAPI lifespan)")

    yield

    # Shutdown
    await service.shutdown()
    logger.info("CountryRiskEvaluatorService shutdown (FastAPI lifespan)")

__all__ = [
    "CountryRiskEvaluatorService",
    "get_service",
    "set_service",
    "reset_service",
    "lifespan",
]
