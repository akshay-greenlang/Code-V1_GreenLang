# -*- coding: utf-8 -*-
"""
SupplierRiskScorerService - Facade for AGENT-EUDR-017

Unified service facade orchestrating all 8 engines of the Supplier Risk
Scorer Agent. Provides a single entry point for supplier risk assessment,
due diligence tracking, documentation analysis, certification validation,
geographic sourcing analysis, network analysis, monitoring/alerting, and
risk report generation.

Engines (8):
    1. SupplierRiskScorer        - Composite supplier risk scoring (Feature 1)
    2. DueDiligenceTracker       - Due diligence tracking & management (Feature 2)
    3. DocumentationAnalyzer     - EUDR documentation analysis (Feature 3)
    4. CertificationValidator    - Certification scheme validation (Feature 4)
    5. GeographicSourcingAnalyzer - Geographic sourcing analysis (Feature 5)
    6. NetworkAnalyzer           - Supplier network analysis (Feature 6)
    7. MonitoringAlertEngine     - Continuous monitoring & alerting (Feature 7)
    8. RiskReportingEngine       - Risk report generation (Feature 8)

Reference Data (3):
    - supplier_risk_database: Sample supplier profiles, benchmarks, peer groups
    - certification_schemes: 8 certification schemes with equivalences
    - document_requirements: EUDR document requirements per commodity

Singleton Pattern:
    Thread-safe singleton with double-checked locking via ``get_service()``.

FastAPI Integration:
    Use the ``lifespan`` async context manager with
    ``FastAPI(lifespan=lifespan)`` for automatic startup/shutdown.

Example:
    >>> from greenlang.agents.eudr.supplier_risk_scorer.setup import (
    ...     SupplierRiskScorerService,
    ...     get_service,
    ... )
    >>> service = get_service()
    >>> await service.startup()
    >>> health = await service.health_check()
    >>> assert health.status == "healthy"
    >>>
    >>> # Full supplier assessment
    >>> assessment = await service.full_assessment(
    ...     supplier_id="SUP-12345",
    ...     include_documentation=True,
    ...     include_certification=True,
    ...     include_geographic=True,
    ...     include_network=True,
    ... )
    >>> assert assessment.risk_level in ["low", "medium", "high", "critical"]
    >>>
    >>> # Track due diligence
    >>> dd_record = await service.track_due_diligence(
    ...     supplier_id="SUP-12345",
    ...     dd_level="enhanced",
    ...     non_conformances=[...],
    ... )
    >>>
    >>> # Compare suppliers
    >>> comparison = await service.compare_suppliers(
    ...     supplier_ids=["SUP-12345", "SUP-67890"],
    ...     metric="risk_score",
    ... )
    >>>
    >>> await service.shutdown()

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-017
Agent ID: GL-EUDR-SRS-017
Regulation: EU 2023/1115 (EUDR) Articles 4, 8, 9, 10, 11, 31
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

from greenlang.agents.eudr.supplier_risk_scorer.config import (
    SupplierRiskScorerConfig,
    get_config,
    set_config,
    reset_config,
)
from greenlang.agents.eudr.supplier_risk_scorer.provenance import (
    ProvenanceTracker,
    get_provenance_tracker,
)
from greenlang.agents.eudr.supplier_risk_scorer.metrics import (
    PROMETHEUS_AVAILABLE,
    record_assessment_completed,
    record_dd_tracked,
    record_documentation_analyzed,
    record_certification_validated,
    record_geographic_analyzed,
    record_network_analyzed,
    record_alert_generated,
    record_report_generated,
    record_api_error,
    observe_assessment_duration,
    observe_dd_duration,
    observe_documentation_duration,
    observe_certification_duration,
    observe_geographic_duration,
    observe_network_duration,
    observe_report_generation_duration,
    set_active_suppliers,
    set_high_risk_suppliers,
    set_pending_dd_actions,
    set_expired_certifications,
    set_active_alerts,
)

# ---------------------------------------------------------------------------
# Internal imports: models
# ---------------------------------------------------------------------------

from greenlang.agents.eudr.supplier_risk_scorer.models import (
    VERSION,
    MAX_RISK_SCORE,
    MIN_RISK_SCORE,
    MAX_BATCH_SIZE,
    EUDR_CUTOFF_DATE,
    EUDR_RETENTION_YEARS,
    SUPPORTED_COMMODITIES,
    SUPPORTED_SCHEMES,
    SUPPORTED_OUTPUT_FORMATS,
    SUPPORTED_REPORT_LANGUAGES,
    DEFAULT_FACTOR_WEIGHTS,
    RiskLevel,
    SupplierType,
    CommodityType,
    CertificationScheme,
    CertificationStatus,
    DocumentType,
    DocumentStatus,
    DDLevel,
    DDStatus,
    NonConformanceType,
    AlertSeverity,
    AlertType,
    ReportType,
    ReportFormat,
    MonitoringFrequency,
    SupplierRiskAssessment,
    DueDiligenceRecord,
    DocumentationProfile,
    CertificationRecord,
    GeographicSourcingProfile,
    SupplierNetwork,
    MonitoringConfig,
    SupplierAlert,
    RiskReport,
    SupplierProfile,
    FactorScore,
    AuditLogEntry,
    AssessSupplierRequest,
    TrackDueDiligenceRequest,
    AnalyzeDocumentationRequest,
    ValidateCertificationRequest,
    AnalyzeGeographicSourcingRequest,
    AnalyzeNetworkRequest,
    ConfigureMonitoringRequest,
    GenerateAlertRequest,
    GenerateReportRequest,
    GetSupplierProfileRequest,
    CompareSupplierRequest,
    GetTrendRequest,
    BatchAssessmentRequest,
    SearchSupplierRequest,
    HealthRequest,
    SupplierRiskResponse,
    DueDiligenceResponse,
    DocumentationResponse,
    CertificationResponse,
    GeographicSourcingResponse,
    NetworkResponse,
    MonitoringResponse,
    AlertResponse,
    ReportResponse,
    ProfileResponse,
    ComparisonResponse,
    TrendResponse,
    BatchResponse,
    SearchResponse,
    HealthResponse,
)

# ---------------------------------------------------------------------------
# Internal imports: 8 engines (lazy imports to avoid circular dependencies)
# ---------------------------------------------------------------------------
# Import engines on-demand in _ensure_engines() to avoid import-time overhead


# ===========================================================================
# Service Facade
# ===========================================================================


class SupplierRiskScorerService:
    """
    Supplier Risk Scorer Service - Orchestrates all 8 engines.

    This service provides a unified interface to the Supplier Risk Scorer
    Agent, composing all 8 engines into a cohesive system for comprehensive
    supplier risk assessment.

    Engines:
        1. SupplierRiskScorer: Composite risk scoring with 8 factors
        2. DueDiligenceTracker: Non-conformance & corrective action tracking
        3. DocumentationAnalyzer: EUDR document validation & completeness
        4. CertificationValidator: Certification scheme & chain-of-custody
        5. GeographicSourcingAnalyzer: Country risk & concentration analysis
        6. NetworkAnalyzer: Multi-tier supplier network risk propagation
        7. MonitoringAlertEngine: Continuous monitoring & watchlist alerts
        8. RiskReportingEngine: Audit-ready report generation

    Thread Safety:
        All public methods are thread-safe. Use get_service() for singleton.

    Attributes:
        config: Agent configuration
        db_pool: PostgreSQL connection pool (lazy-initialized)
        redis_client: Redis async client (lazy-initialized)
        provenance_tracker: Audit trail tracker
    """

    def __init__(self, config: Optional[SupplierRiskScorerConfig] = None) -> None:
        """
        Initialize SupplierRiskScorerService.

        Args:
            config: Configuration (uses global config if None)
        """
        self.config: SupplierRiskScorerConfig = config or get_config()
        self._db_pool: Optional[AsyncConnectionPool] = None
        self._redis_client: Optional[Any] = None
        self._provenance_tracker: ProvenanceTracker = get_provenance_tracker()
        self._startup_time: Optional[datetime] = None
        self._shutdown_requested: bool = False

        # Lazy-loaded engines (initialized on first access)
        self._supplier_risk_scorer: Optional[Any] = None
        self._dd_tracker: Optional[Any] = None
        self._documentation_analyzer: Optional[Any] = None
        self._certification_validator: Optional[Any] = None
        self._geographic_sourcing_analyzer: Optional[Any] = None
        self._network_analyzer: Optional[Any] = None
        self._monitoring_alert_engine: Optional[Any] = None
        self._risk_reporting_engine: Optional[Any] = None

        logger.info(
            f"SupplierRiskScorerService initialized (version={VERSION}, "
            f"db_enabled={self.config.database.enabled}, "
            f"redis_enabled={self.config.redis.enabled})"
        )

    # -----------------------------------------------------------------------
    # Lifecycle management
    # -----------------------------------------------------------------------

    async def startup(self) -> None:
        """
        Start service: initialize database pool, Redis client, engines.

        Raises:
            RuntimeError: If startup fails
        """
        if self._startup_time is not None:
            logger.warning("Service already started, skipping startup")
            return

        logger.info("Starting SupplierRiskScorerService...")
        start_time = time.time()

        try:
            # 1. Initialize database pool
            if self.config.database.enabled and PSYCOPG_POOL_AVAILABLE:
                await self._init_db_pool()
            else:
                logger.info("Database pool disabled or psycopg_pool not available")

            # 2. Initialize Redis client
            if self.config.redis.enabled and REDIS_AVAILABLE:
                await self._init_redis()
            else:
                logger.info("Redis disabled or redis library not available")

            # 3. Engines are lazy-loaded on first access (no startup required)

            self._startup_time = datetime.now(timezone.utc)
            elapsed = time.time() - start_time
            logger.info(
                f"SupplierRiskScorerService started successfully in {elapsed:.2f}s"
            )

        except Exception as e:
            logger.error("Service startup failed: %s", e, exc_info=True)
            raise RuntimeError(f"Service startup failed: {e}") from e

    async def shutdown(self) -> None:
        """
        Shutdown service: close database pool, Redis client.
        """
        if self._shutdown_requested:
            logger.warning("Shutdown already requested, skipping")
            return

        logger.info("Shutting down SupplierRiskScorerService...")
        self._shutdown_requested = True

        try:
            # Close Redis
            if self._redis_client is not None:
                await self._redis_client.close()
                logger.info("Redis client closed")

            # Close database pool
            if self._db_pool is not None:
                await self._db_pool.close()
                logger.info("Database pool closed")

            logger.info("SupplierRiskScorerService shutdown complete")

        except Exception as e:
            logger.error("Shutdown error: %s", e, exc_info=True)

    async def _init_db_pool(self) -> None:
        """Initialize PostgreSQL connection pool."""
        if not PSYCOPG_POOL_AVAILABLE:
            raise RuntimeError("psycopg_pool not available")

        logger.info("Initializing database pool (min=%s, max=%s)", self.config.database.pool_min_size, self.config.database.pool_max_size)
        self._db_pool = AsyncConnectionPool(
            conninfo=self.config.database.url,
            min_size=self.config.database.pool_min_size,
            max_size=self.config.database.pool_max_size,
            timeout=self.config.database.pool_timeout,
            open=False,
        )
        await self._db_pool.open()
        logger.info("Database pool opened")

    async def _init_redis(self) -> None:
        """Initialize Redis async client."""
        if not REDIS_AVAILABLE:
            raise RuntimeError("redis.asyncio not available")

        logger.info("Initializing Redis client")
        self._redis_client = aioredis.from_url(
            self.config.redis.url,
            encoding="utf-8",
            decode_responses=True,
        )
        # Test connection
        await self._redis_client.ping()
        logger.info("Redis client connected")

    def _ensure_engines(self) -> None:
        """
        Lazy-load all 8 engines (import and instantiate on first call).

        This method avoids circular dependencies and import-time overhead.
        """
        if self._supplier_risk_scorer is not None:
            return  # Already initialized

        logger.info("Lazy-loading 8 engines...")

        # Import engines here to avoid import-time overhead
        from greenlang.agents.eudr.supplier_risk_scorer.engines.supplier_risk_scorer import (
            SupplierRiskScorer,
        )
        from greenlang.agents.eudr.supplier_risk_scorer.engines.due_diligence_tracker import (
            DueDiligenceTracker,
        )
        from greenlang.agents.eudr.supplier_risk_scorer.engines.documentation_analyzer import (
            DocumentationAnalyzer,
        )
        from greenlang.agents.eudr.supplier_risk_scorer.engines.certification_validator import (
            CertificationValidator,
        )
        from greenlang.agents.eudr.supplier_risk_scorer.engines.geographic_sourcing_analyzer import (
            GeographicSourcingAnalyzer,
        )
        from greenlang.agents.eudr.supplier_risk_scorer.engines.network_analyzer import (
            NetworkAnalyzer,
        )
        from greenlang.agents.eudr.supplier_risk_scorer.engines.monitoring_alert_engine import (
            MonitoringAlertEngine,
        )
        from greenlang.agents.eudr.supplier_risk_scorer.engines.risk_reporting_engine import (
            RiskReportingEngine,
        )

        # Instantiate engines
        self._supplier_risk_scorer = SupplierRiskScorer(
            config=self.config, db_pool=self._db_pool, redis_client=self._redis_client
        )
        self._dd_tracker = DueDiligenceTracker(
            config=self.config, db_pool=self._db_pool, redis_client=self._redis_client
        )
        self._documentation_analyzer = DocumentationAnalyzer(
            config=self.config, db_pool=self._db_pool, redis_client=self._redis_client
        )
        self._certification_validator = CertificationValidator(
            config=self.config, db_pool=self._db_pool, redis_client=self._redis_client
        )
        self._geographic_sourcing_analyzer = GeographicSourcingAnalyzer(
            config=self.config, db_pool=self._db_pool, redis_client=self._redis_client
        )
        self._network_analyzer = NetworkAnalyzer(
            config=self.config, db_pool=self._db_pool, redis_client=self._redis_client
        )
        self._monitoring_alert_engine = MonitoringAlertEngine(
            config=self.config, db_pool=self._db_pool, redis_client=self._redis_client
        )
        self._risk_reporting_engine = RiskReportingEngine(
            config=self.config, db_pool=self._db_pool, redis_client=self._redis_client
        )

        logger.info("All 8 engines loaded")

    # -----------------------------------------------------------------------
    # Public API: Engine delegation methods
    # -----------------------------------------------------------------------

    async def assess_supplier_risk(
        self,
        request: AssessSupplierRequest,
    ) -> SupplierRiskResponse:
        """
        Assess supplier risk using composite 8-factor scoring.

        Delegates to SupplierRiskScorer engine.

        Args:
            request: Assessment request with supplier_id, factor weights

        Returns:
            SupplierRiskResponse with risk_level, risk_score, factor_scores

        Raises:
            ValueError: If supplier_id invalid or weights invalid
        """
        self._ensure_engines()
        start = time.time()

        try:
            result = await self._supplier_risk_scorer.assess_supplier(request)
            observe_assessment_duration(time.time() - start)
            record_assessment_completed(result.risk_level)
            return result

        except Exception as e:
            record_api_error("assess_supplier_risk")
            logger.error("assess_supplier_risk failed: %s", e, exc_info=True)
            raise

    async def track_due_diligence(
        self,
        request: TrackDueDiligenceRequest,
    ) -> DueDiligenceResponse:
        """
        Track due diligence record with non-conformances & corrective actions.

        Delegates to DueDiligenceTracker engine.

        Args:
            request: DD tracking request with supplier_id, dd_level, non_conformances

        Returns:
            DueDiligenceResponse with dd_status, completion_date, overdue_count

        Raises:
            ValueError: If supplier_id invalid or dd_level invalid
        """
        self._ensure_engines()
        start = time.time()

        try:
            result = await self._dd_tracker.track_due_diligence(request)
            observe_dd_duration(time.time() - start)
            record_dd_tracked(result.dd_level)
            return result

        except Exception as e:
            record_api_error("track_due_diligence")
            logger.error("track_due_diligence failed: %s", e, exc_info=True)
            raise

    async def analyze_documentation(
        self,
        request: AnalyzeDocumentationRequest,
    ) -> DocumentationResponse:
        """
        Analyze EUDR documentation completeness & quality.

        Delegates to DocumentationAnalyzer engine.

        Args:
            request: Documentation analysis request with supplier_id, documents

        Returns:
            DocumentationResponse with completeness_score, missing_documents, expired_documents

        Raises:
            ValueError: If supplier_id invalid or documents invalid
        """
        self._ensure_engines()
        start = time.time()

        try:
            result = await self._documentation_analyzer.analyze_documentation(request)
            observe_documentation_duration(time.time() - start)
            record_documentation_analyzed(result.completeness_score)
            return result

        except Exception as e:
            record_api_error("analyze_documentation")
            logger.error("analyze_documentation failed: %s", e, exc_info=True)
            raise

    async def validate_certification(
        self,
        request: ValidateCertificationRequest,
    ) -> CertificationResponse:
        """
        Validate certification scheme, chain-of-custody, expiry.

        Delegates to CertificationValidator engine.

        Args:
            request: Certification validation request with supplier_id, certifications

        Returns:
            CertificationResponse with valid_certifications, expired_certifications

        Raises:
            ValueError: If supplier_id invalid or certifications invalid
        """
        self._ensure_engines()
        start = time.time()

        try:
            result = await self._certification_validator.validate_certification(request)
            observe_certification_duration(time.time() - start)
            record_certification_validated(result.certification_status)
            return result

        except Exception as e:
            record_api_error("validate_certification")
            logger.error("validate_certification failed: %s", e, exc_info=True)
            raise

    async def analyze_geographic_sourcing(
        self,
        request: AnalyzeGeographicSourcingRequest,
    ) -> GeographicSourcingResponse:
        """
        Analyze geographic sourcing with country risk, concentration, deforestation.

        Delegates to GeographicSourcingAnalyzer engine. Integrates with
        AGENT-EUDR-016 Country Risk Evaluator.

        Args:
            request: Geographic sourcing request with supplier_id, source_countries

        Returns:
            GeographicSourcingResponse with country_risk_scores, concentration_index

        Raises:
            ValueError: If supplier_id invalid or source_countries invalid
        """
        self._ensure_engines()
        start = time.time()

        try:
            result = await self._geographic_sourcing_analyzer.analyze_geographic_sourcing(
                request
            )
            observe_geographic_duration(time.time() - start)
            record_geographic_analyzed(result.concentration_risk_level)
            return result

        except Exception as e:
            record_api_error("analyze_geographic_sourcing")
            logger.error("analyze_geographic_sourcing failed: %s", e, exc_info=True)
            raise

    async def analyze_network(
        self,
        request: AnalyzeNetworkRequest,
    ) -> NetworkResponse:
        """
        Analyze supplier network with multi-tier risk propagation.

        Delegates to NetworkAnalyzer engine.

        Args:
            request: Network analysis request with supplier_id, sub_suppliers

        Returns:
            NetworkResponse with network_risk_score, sub_supplier_risks, circular_dependencies

        Raises:
            ValueError: If supplier_id invalid or sub_suppliers invalid
        """
        self._ensure_engines()
        start = time.time()

        try:
            result = await self._network_analyzer.analyze_network(request)
            observe_network_duration(time.time() - start)
            record_network_analyzed(result.network_risk_level)
            return result

        except Exception as e:
            record_api_error("analyze_network")
            logger.error("analyze_network failed: %s", e, exc_info=True)
            raise

    async def configure_monitoring(
        self,
        request: ConfigureMonitoringRequest,
    ) -> MonitoringResponse:
        """
        Configure continuous monitoring for supplier watchlist.

        Delegates to MonitoringAlertEngine engine.

        Args:
            request: Monitoring config request with supplier_id, frequency, alert_thresholds

        Returns:
            MonitoringResponse with monitoring_id, next_check_date

        Raises:
            ValueError: If supplier_id invalid or frequency invalid
        """
        self._ensure_engines()

        try:
            result = await self._monitoring_alert_engine.configure_monitoring(request)
            return result

        except Exception as e:
            record_api_error("configure_monitoring")
            logger.error("configure_monitoring failed: %s", e, exc_info=True)
            raise

    async def check_alerts(
        self,
        request: GenerateAlertRequest,
    ) -> AlertResponse:
        """
        Check supplier for risk alerts and behavior changes.

        Delegates to MonitoringAlertEngine engine.

        Args:
            request: Alert generation request with supplier_id

        Returns:
            AlertResponse with alerts (severity, type, message)

        Raises:
            ValueError: If supplier_id invalid
        """
        self._ensure_engines()

        try:
            result = await self._monitoring_alert_engine.check_alerts(request)
            if result.alerts:
                for alert in result.alerts:
                    record_alert_generated(alert.severity)
            return result

        except Exception as e:
            record_api_error("check_alerts")
            logger.error("check_alerts failed: %s", e, exc_info=True)
            raise

    async def generate_report(
        self,
        request: GenerateReportRequest,
    ) -> ReportResponse:
        """
        Generate supplier risk report with DDS package & audit trail.

        Delegates to RiskReportingEngine engine.

        Args:
            request: Report generation request with supplier_id, report_type, format

        Returns:
            ReportResponse with report_id, report_url, report_data

        Raises:
            ValueError: If supplier_id invalid or report_type/format invalid
        """
        self._ensure_engines()
        start = time.time()

        try:
            result = await self._risk_reporting_engine.generate_report(request)
            observe_report_generation_duration(time.time() - start)
            record_report_generated(result.report_format)
            return result

        except Exception as e:
            record_api_error("generate_report")
            logger.error("generate_report failed: %s", e, exc_info=True)
            raise

    # -----------------------------------------------------------------------
    # Public API: Cross-engine orchestration methods
    # -----------------------------------------------------------------------

    async def full_assessment(
        self,
        supplier_id: str,
        include_documentation: bool = True,
        include_certification: bool = True,
        include_geographic: bool = True,
        include_network: bool = True,
        include_dd: bool = True,
    ) -> Dict[str, Any]:
        """
        Perform full supplier assessment orchestrating all engines.

        This method calls:
            1. assess_supplier_risk() - Composite risk scoring
            2. analyze_documentation() - Documentation completeness
            3. validate_certification() - Certification validation
            4. analyze_geographic_sourcing() - Country risk & concentration
            5. analyze_network() - Network risk propagation
            6. track_due_diligence() - DD status check

        Args:
            supplier_id: Supplier identifier
            include_documentation: Include documentation analysis
            include_certification: Include certification validation
            include_geographic: Include geographic sourcing analysis
            include_network: Include network analysis
            include_dd: Include DD tracking

        Returns:
            Dict with keys:
                - risk_assessment: SupplierRiskResponse
                - documentation: DocumentationResponse (if included)
                - certification: CertificationResponse (if included)
                - geographic: GeographicSourcingResponse (if included)
                - network: NetworkResponse (if included)
                - due_diligence: DueDiligenceResponse (if included)

        Raises:
            ValueError: If supplier_id invalid
        """
        logger.info("Starting full_assessment for supplier_id=%s", supplier_id)
        start = time.time()

        try:
            # 1. Core risk assessment (always included)
            risk_assessment = await self.assess_supplier_risk(
                AssessSupplierRequest(supplier_id=supplier_id)
            )

            result: Dict[str, Any] = {
                "supplier_id": supplier_id,
                "assessment_timestamp": datetime.now(timezone.utc).isoformat(),
                "risk_assessment": risk_assessment,
            }

            # 2. Documentation analysis (optional)
            if include_documentation:
                documentation = await self.analyze_documentation(
                    AnalyzeDocumentationRequest(supplier_id=supplier_id, documents=[])
                )
                result["documentation"] = documentation

            # 3. Certification validation (optional)
            if include_certification:
                certification = await self.validate_certification(
                    ValidateCertificationRequest(supplier_id=supplier_id, certifications=[])
                )
                result["certification"] = certification

            # 4. Geographic sourcing analysis (optional)
            if include_geographic:
                geographic = await self.analyze_geographic_sourcing(
                    AnalyzeGeographicSourcingRequest(
                        supplier_id=supplier_id, source_countries=[]
                    )
                )
                result["geographic"] = geographic

            # 5. Network analysis (optional)
            if include_network:
                network = await self.analyze_network(
                    AnalyzeNetworkRequest(supplier_id=supplier_id, sub_suppliers=[])
                )
                result["network"] = network

            # 6. Due diligence tracking (optional)
            if include_dd:
                dd = await self.track_due_diligence(
                    TrackDueDiligenceRequest(
                        supplier_id=supplier_id,
                        dd_level=DDLevel.STANDARD,
                        non_conformances=[],
                    )
                )
                result["due_diligence"] = dd

            elapsed = time.time() - start
            logger.info(
                f"full_assessment completed for supplier_id={supplier_id} in {elapsed:.2f}s"
            )

            return result

        except Exception as e:
            record_api_error("full_assessment")
            logger.error("full_assessment failed: %s", e, exc_info=True)
            raise

    async def compare_suppliers(
        self,
        request: CompareSupplierRequest,
    ) -> ComparisonResponse:
        """
        Compare risk scores and metrics across multiple suppliers.

        Args:
            request: Comparison request with supplier_ids, metric

        Returns:
            ComparisonResponse with comparative rankings, delta analysis

        Raises:
            ValueError: If supplier_ids invalid or metric invalid
        """
        self._ensure_engines()
        logger.info("Comparing %s suppliers on metric=%s", len(request.supplier_ids), request.metric)

        try:
            # Assess all suppliers in parallel
            tasks = [
                self.assess_supplier_risk(AssessSupplierRequest(supplier_id=sid))
                for sid in request.supplier_ids
            ]
            assessments = await asyncio.gather(*tasks)

            # Build comparison response
            supplier_scores = [
                {
                    "supplier_id": sid,
                    "risk_score": assessment.risk_score,
                    "risk_level": assessment.risk_level,
                }
                for sid, assessment in zip(request.supplier_ids, assessments)
            ]

            # Sort by risk score (descending = highest risk first)
            supplier_scores.sort(key=lambda x: x["risk_score"], reverse=True)

            return ComparisonResponse(
                comparison_id=str(uuid.uuid4()),
                supplier_count=len(request.supplier_ids),
                metric=request.metric,
                supplier_scores=supplier_scores,
                timestamp=datetime.now(timezone.utc),
            )

        except Exception as e:
            record_api_error("compare_suppliers")
            logger.error("compare_suppliers failed: %s", e, exc_info=True)
            raise

    async def get_supplier_profile(
        self,
        request: GetSupplierProfileRequest,
    ) -> ProfileResponse:
        """
        Retrieve complete supplier profile with historical assessments.

        Args:
            request: Profile request with supplier_id

        Returns:
            ProfileResponse with supplier profile, assessment history

        Raises:
            ValueError: If supplier_id invalid
        """
        self._ensure_engines()
        logger.info("Retrieving profile for supplier_id=%s", request.supplier_id)

        try:
            # Delegate to SupplierRiskScorer for profile retrieval
            result = await self._supplier_risk_scorer.get_supplier_profile(request)
            return result

        except Exception as e:
            record_api_error("get_supplier_profile")
            logger.error("get_supplier_profile failed: %s", e, exc_info=True)
            raise

    async def get_trend(
        self,
        request: GetTrendRequest,
    ) -> TrendResponse:
        """
        Retrieve risk score trend for supplier over time.

        Args:
            request: Trend request with supplier_id, window_months

        Returns:
            TrendResponse with trend data points, trend_direction

        Raises:
            ValueError: If supplier_id invalid
        """
        self._ensure_engines()
        logger.info(
            f"Retrieving trend for supplier_id={request.supplier_id}, window={request.window_months} months"
        )

        try:
            # Delegate to SupplierRiskScorer for trend retrieval
            result = await self._supplier_risk_scorer.get_trend(request)
            return result

        except Exception as e:
            record_api_error("get_trend")
            logger.error("get_trend failed: %s", e, exc_info=True)
            raise

    async def batch_assessment(
        self,
        request: BatchAssessmentRequest,
    ) -> BatchResponse:
        """
        Assess risk for batch of suppliers (up to 500).

        Args:
            request: Batch request with supplier_ids

        Returns:
            BatchResponse with batch_id, results list

        Raises:
            ValueError: If batch size exceeds MAX_BATCH_SIZE
        """
        if len(request.supplier_ids) > MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(request.supplier_ids)} exceeds max {MAX_BATCH_SIZE}"
            )

        logger.info("Starting batch assessment for %s suppliers", len(request.supplier_ids))
        start = time.time()

        try:
            # Assess all suppliers in parallel
            tasks = [
                self.assess_supplier_risk(AssessSupplierRequest(supplier_id=sid))
                for sid in request.supplier_ids
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Separate successes and failures
            successes = []
            failures = []
            for sid, result in zip(request.supplier_ids, results):
                if isinstance(result, Exception):
                    failures.append({"supplier_id": sid, "error": str(result)})
                else:
                    successes.append(result)

            elapsed = time.time() - start
            logger.info(
                f"Batch assessment completed: {len(successes)} success, {len(failures)} failure in {elapsed:.2f}s"
            )

            return BatchResponse(
                batch_id=str(uuid.uuid4()),
                supplier_count=len(request.supplier_ids),
                success_count=len(successes),
                failure_count=len(failures),
                results=successes,
                failures=failures,
                elapsed_seconds=elapsed,
                timestamp=datetime.now(timezone.utc),
            )

        except Exception as e:
            record_api_error("batch_assessment")
            logger.error("batch_assessment failed: %s", e, exc_info=True)
            raise

    async def search_suppliers(
        self,
        request: SearchSupplierRequest,
    ) -> SearchResponse:
        """
        Search suppliers by filters (risk_level, commodity, country, etc).

        Args:
            request: Search request with filters

        Returns:
            SearchResponse with matching suppliers

        Raises:
            ValueError: If filters invalid
        """
        self._ensure_engines()
        logger.info("Searching suppliers with filters: %s", request.filters)

        try:
            # Delegate to SupplierRiskScorer for search
            result = await self._supplier_risk_scorer.search_suppliers(request)
            return result

        except Exception as e:
            record_api_error("search_suppliers")
            logger.error("search_suppliers failed: %s", e, exc_info=True)
            raise

    # -----------------------------------------------------------------------
    # Health check & statistics
    # -----------------------------------------------------------------------

    async def health_check(self) -> HealthResponse:
        """
        Health check aggregating all engine statuses.

        Returns:
            HealthResponse with overall status and per-engine health
        """
        logger.debug("Performing health check")

        try:
            # Check database
            db_healthy = False
            if self._db_pool is not None:
                try:
                    async with self._db_pool.connection() as conn:
                        await conn.execute("SELECT 1")
                        db_healthy = True
                except Exception as e:
                    logger.warning("Database health check failed: %s", e)

            # Check Redis
            redis_healthy = False
            if self._redis_client is not None:
                try:
                    await self._redis_client.ping()
                    redis_healthy = True
                except Exception as e:
                    logger.warning("Redis health check failed: %s", e)

            # Overall status
            overall_status = "healthy" if (db_healthy or not self.config.database.enabled) and (redis_healthy or not self.config.redis.enabled) else "degraded"

            return HealthResponse(
                status=overall_status,
                version=VERSION,
                uptime_seconds=(
                    (datetime.now(timezone.utc) - self._startup_time).total_seconds()
                    if self._startup_time
                    else 0.0
                ),
                database_healthy=db_healthy,
                redis_healthy=redis_healthy,
                engines_loaded=(self._supplier_risk_scorer is not None),
                timestamp=datetime.now(timezone.utc),
            )

        except Exception as e:
            logger.error("health_check failed: %s", e, exc_info=True)
            return HealthResponse(
                status="unhealthy",
                version=VERSION,
                uptime_seconds=0.0,
                database_healthy=False,
                redis_healthy=False,
                engines_loaded=False,
                timestamp=datetime.now(timezone.utc),
            )

    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get service-wide statistics (supplier counts, risk distribution, etc).

        Returns:
            Dict with keys:
                - total_suppliers: int
                - suppliers_by_risk_level: Dict[str, int]
                - pending_dd_actions: int
                - expired_certifications: int
                - active_alerts: int
        """
        self._ensure_engines()
        logger.debug("Retrieving service statistics")

        try:
            # Aggregate statistics from engines
            # NOTE: This is a placeholder - actual implementation would query database
            stats = {
                "total_suppliers": 0,
                "suppliers_by_risk_level": {
                    "low": 0,
                    "medium": 0,
                    "high": 0,
                    "critical": 0,
                },
                "pending_dd_actions": 0,
                "expired_certifications": 0,
                "active_alerts": 0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # TODO: Query database for actual statistics
            # For now, return placeholder
            return stats

        except Exception as e:
            logger.error("get_statistics failed: %s", e, exc_info=True)
            raise


# ===========================================================================
# Singleton management (thread-safe)
# ===========================================================================

_service_instance: Optional[SupplierRiskScorerService] = None
_service_lock: threading.Lock = threading.Lock()


def get_service(config: Optional[SupplierRiskScorerConfig] = None) -> SupplierRiskScorerService:
    """
    Get or create singleton SupplierRiskScorerService instance (thread-safe).

    Args:
        config: Configuration (uses global config if None)

    Returns:
        SupplierRiskScorerService singleton
    """
    global _service_instance

    if _service_instance is None:
        with _service_lock:
            # Double-checked locking
            if _service_instance is None:
                _service_instance = SupplierRiskScorerService(config=config)
                logger.info("Created singleton SupplierRiskScorerService instance")

    return _service_instance


def set_service(service: SupplierRiskScorerService) -> None:
    """
    Set singleton SupplierRiskScorerService instance (for testing).

    Args:
        service: Service instance
    """
    global _service_instance
    with _service_lock:
        _service_instance = service
        logger.info("Set singleton SupplierRiskScorerService instance")


def reset_service() -> None:
    """
    Reset singleton SupplierRiskScorerService instance (for testing).
    """
    global _service_instance
    with _service_lock:
        _service_instance = None
        logger.info("Reset singleton SupplierRiskScorerService instance")


# ===========================================================================
# FastAPI lifespan integration
# ===========================================================================


@asynccontextmanager
async def lifespan(app: Any) -> AsyncIterator[None]:
    """
    FastAPI lifespan async context manager for automatic startup/shutdown.

    Example:
        >>> from fastapi import FastAPI
        >>> from greenlang.agents.eudr.supplier_risk_scorer.setup import lifespan
        >>>
        >>> app = FastAPI(lifespan=lifespan)

    Args:
        app: FastAPI application instance

    Yields:
        None during application lifetime
    """
    # Startup
    service = get_service()
    await service.startup()
    logger.info("SupplierRiskScorerService lifespan started")

    try:
        yield
    finally:
        # Shutdown
        await service.shutdown()
        logger.info("SupplierRiskScorerService lifespan ended")


# ===========================================================================
# Module exports
# ===========================================================================

__all__ = [
    "SupplierRiskScorerService",
    "get_service",
    "set_service",
    "reset_service",
    "lifespan",
    "PSYCOPG_POOL_AVAILABLE",
    "PSYCOPG_AVAILABLE",
    "REDIS_AVAILABLE",
    "OTEL_AVAILABLE",
]
