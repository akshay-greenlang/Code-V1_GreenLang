# -*- coding: utf-8 -*-
"""
Mitigation Measure Designer Service Facade - AGENT-EUDR-029

High-level service facade that wires together all 7 processing engines
into a single cohesive entry point. Provides the primary API used by
the FastAPI router layer to design mitigation strategies, manage
measure templates, estimate effectiveness, track implementation,
verify risk reduction, orchestrate compliance workflows, and generate
mitigation reports per EUDR Article 11 requirements.

This facade implements the Facade Pattern to hide the complexity
of the 7 internal engines behind a clean, use-case-oriented interface.

Engines (7):
    1. MitigationStrategyDesigner      - Strategy design from risk triggers
    2. MeasureTemplateLibrary          - Template catalog management
    3. EffectivenessEstimator          - Three-scenario projection
    4. MeasureImplementationTracker    - Measure lifecycle tracking
    5. RiskReductionVerifier           - Before/after verification
    6. ComplianceWorkflowEngine        - State machine orchestration
    7. MitigationReportGenerator       - DDS-ready report generation

Service Methods:
    Strategy Design:
        - design_strategy()    -> Design mitigation strategy from risk trigger
        - get_strategy()       -> Retrieve strategy by ID
        - list_strategies()    -> List strategies with filters

    Measure Management:
        - approve_measure()    -> Approve a proposed measure
        - start_measure()      -> Begin measure implementation
        - complete_measure()   -> Mark measure as completed
        - cancel_measure()     -> Cancel a measure with reason
        - add_evidence()       -> Attach evidence to a measure

    Templates:
        - list_templates()     -> List measure templates with filters
        - get_template()       -> Retrieve template by ID

    Verification:
        - verify_risk_reduction() -> Verify risk reduction for a strategy

    Reports:
        - generate_report()    -> Generate mitigation report

    Workflows:
        - initiate_workflow()  -> Start a complete mitigation workflow
        - get_workflow_status() -> Get workflow status

    Health:
        - health_check()       -> Component health status

Singleton Pattern:
    Thread-safe singleton with double-checked locking via ``get_service()``.

FastAPI Integration:
    Use the ``lifespan`` async context manager with
    ``FastAPI(lifespan=lifespan)`` for automatic startup/shutdown.

Example:
    >>> from greenlang.agents.eudr.mitigation_measure_designer.setup import (
    ...     MitigationMeasureDesignerService,
    ...     get_service,
    ... )
    >>> service = get_service()
    >>> await service.startup()
    >>> health = await service.health_check()
    >>> assert health["status"] == "healthy"
    >>>
    >>> # Design mitigation strategy
    >>> strategy = await service.design_strategy(risk_trigger)
    >>>
    >>> # Verify risk reduction
    >>> report = await service.verify_risk_reduction(strategy.strategy_id)
    >>>
    >>> await service.shutdown()

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-029 Mitigation Measure Designer (GL-EUDR-MMD-029)
Regulation: EU 2023/1115 (EUDR) Articles 10, 11, 12, 13, 14-16, 29, 31
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
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_VERSION = "1.0.0"
_AGENT_ID = "GL-EUDR-MMD-029"

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

# ---------------------------------------------------------------------------
# Internal imports: config
# ---------------------------------------------------------------------------

from greenlang.agents.eudr.mitigation_measure_designer.config import (
    MitigationMeasureDesignerConfig,
    get_config,
)

# ---------------------------------------------------------------------------
# Provenance import (graceful fallback)
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.mitigation_measure_designer.provenance import (
        ProvenanceTracker,
        GENESIS_HASH,
    )
except ImportError:
    ProvenanceTracker = None  # type: ignore[misc,assignment]
    GENESIS_HASH = (  # type: ignore[assignment]
        "0000000000000000000000000000000000000000000000000000000000000000"
    )

# ---------------------------------------------------------------------------
# Metrics import (graceful fallback)
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.mitigation_measure_designer.metrics import (
        record_strategy_designed,
        record_measure_proposed,
        record_measure_approved,
        record_measure_completed,
        record_verification_run,
        record_report_generated,
        record_workflow_initiated,
        record_api_error,
        observe_strategy_design_duration,
        observe_effectiveness_estimation_duration,
        observe_verification_duration,
        observe_report_generation_duration,
        set_active_strategies,
        set_active_measures,
        set_active_workflows,
        set_templates_loaded,
        set_total_risk_reduction,
        set_pending_verifications,
    )

    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    record_strategy_designed = None  # type: ignore[assignment]
    record_measure_proposed = None  # type: ignore[assignment]
    record_measure_approved = None  # type: ignore[assignment]
    record_measure_completed = None  # type: ignore[assignment]
    record_verification_run = None  # type: ignore[assignment]
    record_report_generated = None  # type: ignore[assignment]
    record_workflow_initiated = None  # type: ignore[assignment]
    record_api_error = None  # type: ignore[assignment]
    observe_strategy_design_duration = None  # type: ignore[assignment]
    observe_effectiveness_estimation_duration = None  # type: ignore[assignment]
    observe_verification_duration = None  # type: ignore[assignment]
    observe_report_generation_duration = None  # type: ignore[assignment]
    set_active_strategies = None  # type: ignore[assignment]
    set_active_measures = None  # type: ignore[assignment]
    set_active_workflows = None  # type: ignore[assignment]
    set_templates_loaded = None  # type: ignore[assignment]
    set_total_risk_reduction = None  # type: ignore[assignment]
    set_pending_verifications = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Model imports (graceful fallback)
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.mitigation_measure_designer.models import (
        EUDRCommodity,
        RiskLevel,
        Article11Category,
        MeasureStatus,
        WorkflowStatus,
        MeasurePriority,
        EffectivenessLevel,
        VerificationResult,
        RiskDimension,
        EvidenceType,
        HealthStatus,
        RiskTrigger,
        MeasureTemplate,
        MitigationMeasure,
        MitigationStrategy,
        EffectivenessEstimate,
        VerificationReport,
        WorkflowState,
        ImplementationMilestone,
        MeasureEvidence,
        MitigationReport,
    )

    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    EUDRCommodity = None  # type: ignore[misc,assignment]
    RiskLevel = None  # type: ignore[misc,assignment]
    Article11Category = None  # type: ignore[misc,assignment]
    MeasureStatus = None  # type: ignore[misc,assignment]
    WorkflowStatus = None  # type: ignore[misc,assignment]
    MeasurePriority = None  # type: ignore[misc,assignment]
    EffectivenessLevel = None  # type: ignore[misc,assignment]
    VerificationResult = None  # type: ignore[misc,assignment]
    RiskDimension = None  # type: ignore[misc,assignment]
    EvidenceType = None  # type: ignore[misc,assignment]
    HealthStatus = None  # type: ignore[misc,assignment]
    RiskTrigger = None  # type: ignore[misc,assignment]
    MeasureTemplate = None  # type: ignore[misc,assignment]
    MitigationMeasure = None  # type: ignore[misc,assignment]
    MitigationStrategy = None  # type: ignore[misc,assignment]
    EffectivenessEstimate = None  # type: ignore[misc,assignment]
    VerificationReport = None  # type: ignore[misc,assignment]
    WorkflowState = None  # type: ignore[misc,assignment]
    ImplementationMilestone = None  # type: ignore[misc,assignment]
    MeasureEvidence = None  # type: ignore[misc,assignment]
    MitigationReport = None  # type: ignore[misc,assignment]

# ---------------------------------------------------------------------------
# Engine imports (conditional -- engines may not exist yet)
# ---------------------------------------------------------------------------

# ---- Engine 1: Mitigation Strategy Designer ----
try:
    from greenlang.agents.eudr.mitigation_measure_designer.mitigation_strategy_designer import (
        MitigationStrategyDesigner,
    )
except ImportError:
    MitigationStrategyDesigner = None  # type: ignore[misc,assignment]

# ---- Engine 2: Measure Template Library ----
try:
    from greenlang.agents.eudr.mitigation_measure_designer.measure_template_library import (
        MeasureTemplateLibrary,
    )
except ImportError:
    MeasureTemplateLibrary = None  # type: ignore[misc,assignment]

# ---- Engine 3: Effectiveness Estimator ----
try:
    from greenlang.agents.eudr.mitigation_measure_designer.effectiveness_estimator import (
        EffectivenessEstimator,
    )
except ImportError:
    EffectivenessEstimator = None  # type: ignore[misc,assignment]

# ---- Engine 4: Measure Implementation Tracker ----
try:
    from greenlang.agents.eudr.mitigation_measure_designer.measure_implementation_tracker import (
        MeasureImplementationTracker,
    )
except ImportError:
    MeasureImplementationTracker = None  # type: ignore[misc,assignment]

# ---- Engine 5: Risk Reduction Verifier ----
try:
    from greenlang.agents.eudr.mitigation_measure_designer.risk_reduction_verifier import (
        RiskReductionVerifier,
    )
except ImportError:
    RiskReductionVerifier = None  # type: ignore[misc,assignment]

# ---- Engine 6: Compliance Workflow Engine ----
try:
    from greenlang.agents.eudr.mitigation_measure_designer.compliance_workflow_engine import (
        ComplianceWorkflowEngine,
    )
except ImportError:
    ComplianceWorkflowEngine = None  # type: ignore[misc,assignment]

# ---- Engine 7: Mitigation Report Generator ----
try:
    from greenlang.agents.eudr.mitigation_measure_designer.mitigation_report_generator import (
        MitigationReportGenerator,
    )
except ImportError:
    MitigationReportGenerator = None  # type: ignore[misc,assignment]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with second precision."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute deterministic SHA-256 hash for provenance.

    Args:
        data: JSON-serializable object.

    Returns:
        64-character lowercase hex SHA-256 hash string.
    """
    canonical = json.dumps(
        data, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _safe_record(metric_fn: Any, *args: Any) -> None:
    """Safely call a metrics function if available.

    Args:
        metric_fn: Metrics function (may be None).
        *args: Arguments to pass.
    """
    if metric_fn is not None:
        try:
            metric_fn(*args)
        except Exception:
            pass


def _safe_observe(metric_fn: Any, value: float) -> None:
    """Safely observe a histogram metric if available.

    Args:
        metric_fn: Histogram observe function (may be None).
        value: Duration in seconds to observe.
    """
    if metric_fn is not None:
        try:
            metric_fn(value)
        except Exception:
            pass


def _safe_gauge(metric_fn: Any, value: Any) -> None:
    """Safely set a gauge metric if available.

    Args:
        metric_fn: Gauge set function (may be None).
        value: Value to set.
    """
    if metric_fn is not None:
        try:
            metric_fn(value)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Service Facade
# ---------------------------------------------------------------------------


class MitigationMeasureDesignerService:
    """Unified service facade for AGENT-EUDR-029.

    Aggregates all 7 processing engines and provides a clean API for
    mitigation measure design, implementation tracking, risk reduction
    verification, and compliance reporting per EUDR Article 11.

    This class manages in-memory storage for strategies, measures,
    workflows, and evidence. In production, these are backed by
    PostgreSQL persistence via the engine layer.

    Attributes:
        config: Agent configuration.
        _strategy_designer: Engine 1 -- strategy design.
        _template_library: Engine 2 -- measure templates.
        _effectiveness_estimator: Engine 3 -- effectiveness estimation.
        _implementation_tracker: Engine 4 -- implementation tracking.
        _risk_reduction_verifier: Engine 5 -- risk reduction verification.
        _compliance_workflow: Engine 6 -- compliance workflow state machine.
        _report_generator: Engine 7 -- report generation.
        _provenance: SHA-256 provenance tracker.
        _initialized: Whether startup has completed.

    Example:
        >>> service = MitigationMeasureDesignerService()
        >>> await service.startup()
        >>> health = await service.health_check()
        >>> assert health["status"] == "healthy"
    """

    def __init__(
        self,
        config: Optional[MitigationMeasureDesignerConfig] = None,
    ) -> None:
        """Initialize the service facade.

        Args:
            config: Optional configuration override.
                   If None, uses get_config() singleton.
        """
        self.config = config or get_config()

        # Provenance tracker
        if ProvenanceTracker is not None:
            self._provenance = ProvenanceTracker()
        else:
            self._provenance = None

        # Database / cache handles
        self._db_pool: Optional[Any] = None
        self._redis: Optional[Any] = None

        # Engine references (lazy initialized in startup)
        self._strategy_designer: Optional[Any] = None
        self._template_library: Optional[Any] = None
        self._effectiveness_estimator: Optional[Any] = None
        self._implementation_tracker: Optional[Any] = None
        self._risk_reduction_verifier: Optional[Any] = None
        self._compliance_workflow: Optional[Any] = None
        self._report_generator: Optional[Any] = None
        self._engines: Dict[str, Any] = {}

        # In-memory stores (used when engines are unavailable)
        self._strategies: Dict[str, Any] = {}
        self._measures: Dict[str, Any] = {}
        self._workflows: Dict[str, Any] = {}
        self._evidence: Dict[str, List[Any]] = {}
        self._templates: List[Any] = []
        self._verification_reports: Dict[str, Any] = {}
        self._reports: Dict[str, Any] = {}

        self._initialized = False

        logger.info("MitigationMeasureDesignerService created")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        """Initialize all engines and external connections.

        Performs database pool creation, Redis connection, engine
        initialization, and reference data loading. Logs startup
        time and engine availability.

        Raises:
            RuntimeError: If critical engine initialization fails.
        """
        start = time.monotonic()
        logger.info("MitigationMeasureDesignerService startup initiated")

        # Initialize database pool
        if PSYCOPG_POOL_AVAILABLE:
            try:
                db_url = (
                    f"host={self.config.db_host} port={self.config.db_port} "
                    f"dbname={self.config.db_name} user={self.config.db_user} "
                    f"password={self.config.db_password}"
                )
                self._db_pool = AsyncConnectionPool(
                    conninfo=db_url,
                    min_size=self.config.db_pool_min,
                    max_size=self.config.db_pool_max,
                    open=False,
                )
                await self._db_pool.open()
                logger.info("PostgreSQL connection pool opened")
            except Exception as e:
                logger.warning(f"PostgreSQL pool init failed: {e}")
                self._db_pool = None

        # Initialize Redis
        if REDIS_AVAILABLE:
            try:
                redis_url = (
                    f"redis://{self.config.redis_host}:"
                    f"{self.config.redis_port}/{self.config.redis_db}"
                )
                self._redis = aioredis.from_url(
                    redis_url,
                    decode_responses=True,
                )
                await self._redis.ping()
                logger.info("Redis connection established")
            except Exception as e:
                logger.warning(f"Redis init failed: {e}")
                self._redis = None

        # Initialize engines
        self._init_engines()

        # Load default templates if template library available
        if self._template_library is not None:
            try:
                if hasattr(self._template_library, "load_templates"):
                    self._template_library.load_templates()
                    template_count = self._template_library.template_count
                    _safe_gauge(set_templates_loaded, template_count)
                    logger.info(
                        f"Template library loaded: {template_count} templates"
                    )
            except Exception as e:
                logger.warning(f"Template library load failed: {e}")

        self._initialized = True
        elapsed = (time.monotonic() - start) * 1000
        engine_count = len(self._engines)

        # Record startup provenance
        if self._provenance is not None:
            self._provenance.record(
                entity_type="service",
                action="startup",
                entity_id=_AGENT_ID,
                actor="system",
                metadata={
                    "engines_loaded": engine_count,
                    "startup_time_ms": round(elapsed, 2),
                    "db_available": self._db_pool is not None,
                    "redis_available": self._redis is not None,
                },
            )

        logger.info(
            f"MitigationMeasureDesignerService startup complete: "
            f"{engine_count}/7 engines in {elapsed:.1f}ms"
        )

    def _init_engines(self) -> None:
        """Initialize all 7 processing engines with graceful degradation.

        Each engine is initialized independently so that failures in one
        engine do not prevent other engines from loading.
        """
        engine_specs: List[Tuple[str, Any]] = [
            ("mitigation_strategy_designer", MitigationStrategyDesigner),
            ("measure_template_library", MeasureTemplateLibrary),
            ("effectiveness_estimator", EffectivenessEstimator),
            ("measure_implementation_tracker", MeasureImplementationTracker),
            ("risk_reduction_verifier", RiskReductionVerifier),
            ("compliance_workflow_engine", ComplianceWorkflowEngine),
            ("mitigation_report_generator", MitigationReportGenerator),
        ]

        for name, engine_cls in engine_specs:
            if engine_cls is not None:
                try:
                    engine = engine_cls(config=self.config)
                    self._engines[name] = engine
                    logger.info(f"Engine '{name}' initialized")
                except Exception as e:
                    logger.warning(f"Engine '{name}' init failed: {e}")
            else:
                logger.debug(f"Engine '{name}' class not available")

        # Wire up convenience references
        self._strategy_designer = self._engines.get(
            "mitigation_strategy_designer"
        )
        self._template_library = self._engines.get(
            "measure_template_library"
        )
        self._effectiveness_estimator = self._engines.get(
            "effectiveness_estimator"
        )
        self._implementation_tracker = self._engines.get(
            "measure_implementation_tracker"
        )
        self._risk_reduction_verifier = self._engines.get(
            "risk_reduction_verifier"
        )
        self._compliance_workflow = self._engines.get(
            "compliance_workflow_engine"
        )
        self._report_generator = self._engines.get(
            "mitigation_report_generator"
        )

    async def shutdown(self) -> None:
        """Gracefully shutdown all connections and engines.

        Closes database pool, Redis connection, and any
        engine-specific resources.
        """
        logger.info("MitigationMeasureDesignerService shutdown initiated")

        # Shutdown engines with async shutdown methods
        for name, engine in self._engines.items():
            if hasattr(engine, "shutdown"):
                try:
                    result = engine.shutdown()
                    if asyncio.iscoroutine(result):
                        await result
                    logger.info(f"Engine '{name}' shut down")
                except Exception as e:
                    logger.warning(f"Engine '{name}' shutdown error: {e}")

        # Close Redis
        if self._redis is not None:
            try:
                await self._redis.close()
                logger.info("Redis connection closed")
            except Exception as e:
                logger.warning(f"Redis close error: {e}")

        # Close database pool
        if self._db_pool is not None:
            try:
                await self._db_pool.close()
                logger.info("PostgreSQL pool closed")
            except Exception as e:
                logger.warning(f"PostgreSQL pool close error: {e}")

        # Record shutdown provenance
        if self._provenance is not None:
            self._provenance.record(
                entity_type="service",
                action="shutdown",
                entity_id=_AGENT_ID,
                actor="system",
            )

        self._initialized = False
        logger.info("MitigationMeasureDesignerService shutdown complete")

    # ------------------------------------------------------------------
    # Strategy Design
    # ------------------------------------------------------------------

    async def design_strategy(
        self,
        risk_trigger: Any,
    ) -> Any:
        """Design a mitigation strategy for a risk trigger.

        Analyses the risk trigger, selects appropriate measure templates,
        estimates effectiveness, and produces a structured strategy with
        prioritized measures targeting the configured risk reduction goal.

        Args:
            risk_trigger: RiskTrigger model with risk source details
                including operator_id, commodity, risk_level, risk_score,
                risk_dimension, and country_codes.

        Returns:
            MitigationStrategy with designed measures and estimates.

        Raises:
            ValueError: If the risk trigger is invalid or incomplete.
        """
        start = time.monotonic()
        strategy_id = _new_uuid()

        logger.info(
            f"Designing mitigation strategy {strategy_id} for "
            f"operator={risk_trigger.operator_id}, "
            f"commodity={risk_trigger.commodity.value}, "
            f"risk_level={risk_trigger.risk_level.value}"
        )

        # Delegate to engine if available
        if self._strategy_designer is not None:
            try:
                strategy = await self._strategy_designer.design_strategy(
                    risk_trigger=risk_trigger,
                )
                self._strategies[strategy.strategy_id] = strategy
                _safe_record(record_strategy_designed, risk_trigger.commodity.value)
                elapsed = time.monotonic() - start
                _safe_observe(observe_strategy_design_duration, elapsed)
                _safe_gauge(set_active_strategies, len(self._strategies))
                return strategy
            except Exception as e:
                logger.warning(
                    f"Strategy designer engine failed, using fallback: {e}"
                )

        # Fallback: in-memory strategy creation
        strategy = self._create_fallback_strategy(strategy_id, risk_trigger)
        self._strategies[strategy_id] = strategy

        # Record provenance
        if self._provenance is not None:
            self._provenance.record(
                entity_type="strategy",
                action="create",
                entity_id=strategy_id,
                actor=risk_trigger.operator_id,
                metadata={
                    "commodity": risk_trigger.commodity.value,
                    "risk_level": risk_trigger.risk_level.value,
                    "risk_score": str(risk_trigger.risk_score),
                    "measure_count": len(strategy.get("measures", [])),
                },
            )

        _safe_record(record_strategy_designed, risk_trigger.commodity.value)
        elapsed = time.monotonic() - start
        _safe_observe(observe_strategy_design_duration, elapsed)
        _safe_gauge(set_active_strategies, len(self._strategies))

        logger.info(
            f"Strategy {strategy_id} designed in {elapsed * 1000:.1f}ms "
            f"with {len(strategy.get('measures', []))} measures"
        )

        return strategy

    def _create_fallback_strategy(
        self,
        strategy_id: str,
        risk_trigger: Any,
    ) -> Dict[str, Any]:
        """Create a fallback in-memory strategy when engines are unavailable.

        Args:
            strategy_id: Generated strategy identifier.
            risk_trigger: RiskTrigger model.

        Returns:
            Dictionary representing the strategy.
        """
        now = _utcnow()
        target_score = self.config.mitigation_target_score
        risk_score = risk_trigger.risk_score

        # Calculate required reduction
        required_reduction = max(Decimal("0"), risk_score - target_score)

        # Select template-based measures
        measures = self._select_fallback_measures(
            risk_trigger, required_reduction
        )

        strategy = {
            "strategy_id": strategy_id,
            "operator_id": risk_trigger.operator_id,
            "commodity": risk_trigger.commodity.value,
            "risk_level": risk_trigger.risk_level.value,
            "risk_dimension": risk_trigger.risk_dimension.value,
            "current_risk_score": str(risk_score),
            "target_risk_score": str(target_score),
            "required_reduction": str(required_reduction),
            "measures": measures,
            "status": "proposed",
            "created_at": now.isoformat(),
            "provenance_hash": _compute_hash({
                "strategy_id": strategy_id,
                "operator_id": risk_trigger.operator_id,
                "commodity": risk_trigger.commodity.value,
                "risk_score": str(risk_score),
                "created_at": now.isoformat(),
            }),
        }

        # Track individual measures
        for measure in measures:
            measure_id = measure["measure_id"]
            self._measures[measure_id] = measure
            _safe_record(
                record_measure_proposed,
                risk_trigger.commodity.value,
            )

        return strategy

    def _select_fallback_measures(
        self,
        risk_trigger: Any,
        required_reduction: Decimal,
    ) -> List[Dict[str, Any]]:
        """Select fallback measures based on risk level and dimension.

        Args:
            risk_trigger: RiskTrigger model.
            required_reduction: Required risk score reduction.

        Returns:
            List of measure dictionaries.
        """
        measures: List[Dict[str, Any]] = []
        now = _utcnow()
        dimension = risk_trigger.risk_dimension.value
        commodity = risk_trigger.commodity.value

        # Standard measure catalog per dimension
        dimension_measures: Dict[str, List[Dict[str, str]]] = {
            "country": [
                {
                    "title": "Enhanced country-level monitoring",
                    "category": "monitoring_enhancement",
                    "reduction": "10",
                },
                {
                    "title": "Alternative sourcing country evaluation",
                    "category": "supply_chain_restructuring",
                    "reduction": "15",
                },
                {
                    "title": "Country compliance certification requirement",
                    "category": "certification_requirement",
                    "reduction": "12",
                },
            ],
            "supplier": [
                {
                    "title": "Supplier due diligence enhancement",
                    "category": "supplier_engagement",
                    "reduction": "12",
                },
                {
                    "title": "Supplier capacity building program",
                    "category": "capacity_building",
                    "reduction": "10",
                },
                {
                    "title": "Contractual compliance clauses",
                    "category": "contractual_safeguard",
                    "reduction": "8",
                },
            ],
            "commodity": [
                {
                    "title": "Commodity traceability enhancement",
                    "category": "monitoring_enhancement",
                    "reduction": "10",
                },
                {
                    "title": "Certified commodity sourcing",
                    "category": "certification_requirement",
                    "reduction": "15",
                },
                {
                    "title": "Mass balance verification protocol",
                    "category": "supply_chain_restructuring",
                    "reduction": "8",
                },
            ],
            "corruption": [
                {
                    "title": "Third-party anti-corruption audit",
                    "category": "certification_requirement",
                    "reduction": "12",
                },
                {
                    "title": "Governance transparency program",
                    "category": "supplier_engagement",
                    "reduction": "10",
                },
            ],
            "deforestation": [
                {
                    "title": "Satellite monitoring deployment",
                    "category": "monitoring_enhancement",
                    "reduction": "15",
                },
                {
                    "title": "Zero-deforestation commitment enforcement",
                    "category": "contractual_safeguard",
                    "reduction": "12",
                },
                {
                    "title": "Reforestation partnership program",
                    "category": "capacity_building",
                    "reduction": "8",
                },
            ],
            "environmental": [
                {
                    "title": "Environmental impact assessment",
                    "category": "monitoring_enhancement",
                    "reduction": "10",
                },
                {
                    "title": "Sustainable land use certification",
                    "category": "certification_requirement",
                    "reduction": "12",
                },
            ],
        }

        templates = dimension_measures.get(dimension, [])
        accumulated = Decimal("0")
        priority_order = 1

        for tmpl in templates:
            if accumulated >= required_reduction:
                break

            measure_id = _new_uuid()
            reduction = Decimal(tmpl["reduction"])
            deadline = now + timedelta(
                days=self.config.default_deadline_days
            )

            measure = {
                "measure_id": measure_id,
                "strategy_id": "",
                "title": tmpl["title"],
                "description": (
                    f"{tmpl['title']} for {commodity} sourcing "
                    f"addressing {dimension} risk dimension."
                ),
                "category": tmpl["category"],
                "commodity": commodity,
                "risk_dimension": dimension,
                "priority": priority_order,
                "status": "proposed",
                "estimated_reduction": str(reduction),
                "deadline": deadline.isoformat(),
                "created_at": now.isoformat(),
            }

            measures.append(measure)
            accumulated += reduction
            priority_order += 1

        return measures

    async def get_strategy(
        self,
        strategy_id: str,
    ) -> Optional[Any]:
        """Retrieve a strategy by its identifier.

        Args:
            strategy_id: Strategy identifier.

        Returns:
            Strategy data or None if not found.
        """
        # Check engine first
        if self._strategy_designer is not None:
            try:
                return await self._strategy_designer.get_strategy(strategy_id)
            except Exception as e:
                logger.debug(f"Engine strategy lookup failed: {e}")

        # Fallback to in-memory
        return self._strategies.get(strategy_id)

    async def list_strategies(
        self,
        operator_id: Optional[str] = None,
        commodity: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Any]:
        """List strategies with optional filters.

        Args:
            operator_id: Filter by operator identifier.
            commodity: Filter by EUDR commodity.
            status: Filter by strategy status.

        Returns:
            List of matching strategies.
        """
        # Delegate to engine if available
        if self._strategy_designer is not None:
            try:
                return await self._strategy_designer.list_strategies(
                    operator_id=operator_id,
                    commodity=commodity,
                    status=status,
                )
            except Exception as e:
                logger.debug(f"Engine list_strategies failed: {e}")

        # Fallback: filter in-memory strategies
        results = list(self._strategies.values())

        if operator_id:
            results = [
                s for s in results
                if s.get("operator_id") == operator_id
            ]
        if commodity:
            results = [
                s for s in results
                if s.get("commodity") == commodity
            ]
        if status:
            results = [
                s for s in results
                if s.get("status") == status
            ]

        return results

    # ------------------------------------------------------------------
    # Measure Management
    # ------------------------------------------------------------------

    async def approve_measure(
        self,
        measure_id: str,
        approved_by: str,
    ) -> Any:
        """Approve a proposed measure for implementation.

        Transitions a measure from 'proposed' to 'approved' status,
        recording the approver and timestamp for audit trail.

        Args:
            measure_id: Measure identifier.
            approved_by: User or system identifier approving the measure.

        Returns:
            Updated measure data.

        Raises:
            ValueError: If measure not found or not in proposed status.
        """
        logger.info(
            f"Approving measure {measure_id} by {approved_by}"
        )

        # Delegate to engine if available
        if self._implementation_tracker is not None:
            try:
                result = await self._implementation_tracker.approve_measure(
                    measure_id=measure_id,
                    approved_by=approved_by,
                )
                _safe_record(record_measure_approved)
                return result
            except Exception as e:
                logger.debug(f"Engine approve_measure failed: {e}")

        # Fallback: in-memory state transition
        measure = self._measures.get(measure_id)
        if measure is None:
            raise ValueError(f"Measure {measure_id} not found")

        if measure.get("status") != "proposed":
            raise ValueError(
                f"Measure {measure_id} is in '{measure.get('status')}' "
                f"status, expected 'proposed'"
            )

        measure["status"] = "approved"
        measure["approved_by"] = approved_by
        measure["approved_at"] = _utcnow().isoformat()

        # Record provenance
        if self._provenance is not None:
            self._provenance.record(
                entity_type="measure",
                action="approve",
                entity_id=measure_id,
                actor=approved_by,
            )

        _safe_record(record_measure_approved)
        _safe_gauge(set_active_measures, len(self._measures))

        logger.info(f"Measure {measure_id} approved by {approved_by}")
        return measure

    async def start_measure(
        self,
        measure_id: str,
    ) -> Any:
        """Start implementation of an approved measure.

        Transitions a measure from 'approved' to 'in_progress' status.

        Args:
            measure_id: Measure identifier.

        Returns:
            Updated measure data.

        Raises:
            ValueError: If measure not found or not in approved status.
        """
        logger.info(f"Starting measure {measure_id}")

        # Delegate to engine if available
        if self._implementation_tracker is not None:
            try:
                return await self._implementation_tracker.start_measure(
                    measure_id=measure_id,
                )
            except Exception as e:
                logger.debug(f"Engine start_measure failed: {e}")

        # Fallback: in-memory state transition
        measure = self._measures.get(measure_id)
        if measure is None:
            raise ValueError(f"Measure {measure_id} not found")

        if measure.get("status") != "approved":
            raise ValueError(
                f"Measure {measure_id} is in '{measure.get('status')}' "
                f"status, expected 'approved'"
            )

        measure["status"] = "in_progress"
        measure["started_at"] = _utcnow().isoformat()

        # Record provenance
        if self._provenance is not None:
            self._provenance.record(
                entity_type="measure",
                action="start",
                entity_id=measure_id,
                actor="system",
            )

        logger.info(f"Measure {measure_id} started")
        return measure

    async def complete_measure(
        self,
        measure_id: str,
        actual_reduction: Optional[Decimal] = None,
    ) -> Any:
        """Mark a measure as completed.

        Transitions a measure from 'in_progress' to 'completed' status
        and optionally records the actual risk reduction achieved.

        Args:
            measure_id: Measure identifier.
            actual_reduction: Actual risk reduction achieved (optional).

        Returns:
            Updated measure data.

        Raises:
            ValueError: If measure not found or not in_progress.
        """
        logger.info(f"Completing measure {measure_id}")

        # Delegate to engine if available
        if self._implementation_tracker is not None:
            try:
                result = await self._implementation_tracker.complete_measure(
                    measure_id=measure_id,
                    actual_reduction=actual_reduction,
                )
                _safe_record(record_measure_completed)
                return result
            except Exception as e:
                logger.debug(f"Engine complete_measure failed: {e}")

        # Fallback: in-memory state transition
        measure = self._measures.get(measure_id)
        if measure is None:
            raise ValueError(f"Measure {measure_id} not found")

        if measure.get("status") != "in_progress":
            raise ValueError(
                f"Measure {measure_id} is in '{measure.get('status')}' "
                f"status, expected 'in_progress'"
            )

        measure["status"] = "completed"
        measure["completed_at"] = _utcnow().isoformat()
        if actual_reduction is not None:
            measure["actual_reduction"] = str(actual_reduction)

        # Record provenance
        if self._provenance is not None:
            self._provenance.record(
                entity_type="measure",
                action="complete",
                entity_id=measure_id,
                actor="system",
                metadata={
                    "actual_reduction": str(actual_reduction)
                    if actual_reduction
                    else None,
                },
            )

        _safe_record(record_measure_completed)

        logger.info(
            f"Measure {measure_id} completed"
            + (
                f" (actual_reduction={actual_reduction})"
                if actual_reduction
                else ""
            )
        )
        return measure

    async def cancel_measure(
        self,
        measure_id: str,
        reason: str,
    ) -> Any:
        """Cancel a measure with a documented reason.

        Transitions a measure to 'cancelled' status. The reason is
        recorded for audit trail purposes.

        Args:
            measure_id: Measure identifier.
            reason: Cancellation reason text.

        Returns:
            Updated measure data.

        Raises:
            ValueError: If measure not found or already completed/cancelled.
        """
        logger.info(f"Cancelling measure {measure_id}: {reason}")

        # Delegate to engine if available
        if self._implementation_tracker is not None:
            try:
                return await self._implementation_tracker.cancel_measure(
                    measure_id=measure_id,
                    reason=reason,
                )
            except Exception as e:
                logger.debug(f"Engine cancel_measure failed: {e}")

        # Fallback: in-memory state transition
        measure = self._measures.get(measure_id)
        if measure is None:
            raise ValueError(f"Measure {measure_id} not found")

        terminal = {"completed", "cancelled"}
        if measure.get("status") in terminal:
            raise ValueError(
                f"Measure {measure_id} is in terminal state "
                f"'{measure.get('status')}' and cannot be cancelled"
            )

        measure["status"] = "cancelled"
        measure["cancelled_at"] = _utcnow().isoformat()
        measure["cancellation_reason"] = reason

        # Record provenance
        if self._provenance is not None:
            self._provenance.record(
                entity_type="measure",
                action="cancel",
                entity_id=measure_id,
                actor="system",
                metadata={"reason": reason},
            )

        logger.info(f"Measure {measure_id} cancelled: {reason}")
        return measure

    async def add_evidence(
        self,
        measure_id: str,
        evidence_type: str,
        title: str,
        file_ref: str,
        uploaded_by: str,
    ) -> Dict[str, Any]:
        """Attach evidence to a measure for audit trail.

        Creates an evidence record associated with a measure, enabling
        verifiable documentation of mitigation actions taken.

        Args:
            measure_id: Measure identifier.
            evidence_type: Type of evidence (e.g. 'certificate', 'report').
            title: Human-readable evidence title.
            file_ref: File reference or storage path.
            uploaded_by: User who uploaded the evidence.

        Returns:
            Evidence record dictionary.

        Raises:
            ValueError: If measure not found.
        """
        logger.info(
            f"Adding evidence to measure {measure_id}: {title}"
        )

        # Validate measure exists
        measure = self._measures.get(measure_id)
        if measure is None:
            # Check engine
            if self._implementation_tracker is not None:
                try:
                    return await self._implementation_tracker.add_evidence(
                        measure_id=measure_id,
                        evidence_type=evidence_type,
                        title=title,
                        file_ref=file_ref,
                        uploaded_by=uploaded_by,
                    )
                except Exception as e:
                    logger.debug(f"Engine add_evidence failed: {e}")

            raise ValueError(f"Measure {measure_id} not found")

        evidence_id = _new_uuid()
        now = _utcnow()

        evidence_record = {
            "evidence_id": evidence_id,
            "measure_id": measure_id,
            "evidence_type": evidence_type,
            "title": title,
            "file_ref": file_ref,
            "uploaded_by": uploaded_by,
            "uploaded_at": now.isoformat(),
            "hash": _compute_hash({
                "evidence_id": evidence_id,
                "measure_id": measure_id,
                "title": title,
                "file_ref": file_ref,
                "uploaded_at": now.isoformat(),
            }),
        }

        # Store evidence
        if measure_id not in self._evidence:
            self._evidence[measure_id] = []
        self._evidence[measure_id].append(evidence_record)

        # Record provenance
        if self._provenance is not None:
            self._provenance.record(
                entity_type="evidence",
                action="upload",
                entity_id=evidence_id,
                actor=uploaded_by,
                metadata={
                    "measure_id": measure_id,
                    "evidence_type": evidence_type,
                },
            )

        logger.info(
            f"Evidence {evidence_id} added to measure {measure_id}"
        )
        return evidence_record

    # ------------------------------------------------------------------
    # Templates
    # ------------------------------------------------------------------

    async def list_templates(
        self,
        dimension: Optional[str] = None,
        category: Optional[str] = None,
        commodity: Optional[str] = None,
    ) -> List[Any]:
        """List measure templates with optional filters.

        Args:
            dimension: Filter by risk dimension.
            category: Filter by Article 11 category.
            commodity: Filter by EUDR commodity.

        Returns:
            List of matching measure templates.
        """
        # Delegate to engine if available
        if self._template_library is not None:
            try:
                return await self._template_library.list_templates(
                    dimension=dimension,
                    category=category,
                    commodity=commodity,
                )
            except Exception as e:
                logger.debug(f"Engine list_templates failed: {e}")

        # Fallback: return filtered in-memory templates
        results = list(self._templates)

        if dimension:
            results = [
                t for t in results
                if t.get("risk_dimension") == dimension
            ]
        if category:
            results = [
                t for t in results
                if t.get("category") == category
            ]
        if commodity:
            results = [
                t for t in results
                if commodity in t.get("commodities", [])
                or t.get("commodity") == commodity
            ]

        return results

    async def get_template(
        self,
        template_id: str,
    ) -> Optional[Any]:
        """Retrieve a measure template by its identifier.

        Args:
            template_id: Template identifier.

        Returns:
            Template data or None if not found.
        """
        # Delegate to engine if available
        if self._template_library is not None:
            try:
                return await self._template_library.get_template(template_id)
            except Exception as e:
                logger.debug(f"Engine get_template failed: {e}")

        # Fallback: search in-memory templates
        for tmpl in self._templates:
            if tmpl.get("template_id") == template_id:
                return tmpl
        return None

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    async def verify_risk_reduction(
        self,
        strategy_id: str,
    ) -> Dict[str, Any]:
        """Verify risk reduction for a strategy.

        Compares the current risk score against the initial risk score
        at the time the strategy was created, evaluates measure
        effectiveness, and produces a verification report.

        Args:
            strategy_id: Strategy identifier.

        Returns:
            Verification report dictionary.

        Raises:
            ValueError: If strategy not found.
        """
        start = time.monotonic()

        logger.info(
            f"Verifying risk reduction for strategy {strategy_id}"
        )

        # Delegate to engine if available
        if self._risk_reduction_verifier is not None:
            try:
                result = await self._risk_reduction_verifier.verify(
                    strategy_id=strategy_id,
                )
                _safe_record(record_verification_run)
                elapsed = time.monotonic() - start
                _safe_observe(observe_verification_duration, elapsed)
                return result
            except Exception as e:
                logger.debug(f"Engine verify failed: {e}")

        # Fallback: compute from in-memory data
        strategy = self._strategies.get(strategy_id)
        if strategy is None:
            raise ValueError(f"Strategy {strategy_id} not found")

        initial_score = Decimal(strategy.get("current_risk_score", "50"))
        target_score = Decimal(strategy.get("target_risk_score", "30"))

        # Calculate actual reduction from completed measures
        measures = [
            m for m in self._measures.values()
            if m.get("status") == "completed"
        ]

        total_actual_reduction = Decimal("0")
        for m in measures:
            actual = m.get("actual_reduction")
            if actual:
                total_actual_reduction += Decimal(actual)
            else:
                estimated = m.get("estimated_reduction", "0")
                total_actual_reduction += (
                    Decimal(estimated) * self.config.conservative_factor
                )

        current_score = max(
            Decimal("0"),
            initial_score - total_actual_reduction,
        )
        achieved_reduction = initial_score - current_score
        required_reduction = initial_score - target_score
        gap = max(Decimal("0"), target_score - current_score)

        # Determine verification result
        if current_score <= target_score:
            result_status = "verified"
        elif achieved_reduction > Decimal("0"):
            result_status = "partial"
        else:
            result_status = "insufficient"

        report_id = _new_uuid()
        now = _utcnow()

        verification_report = {
            "report_id": report_id,
            "strategy_id": strategy_id,
            "initial_risk_score": str(initial_score),
            "current_risk_score": str(current_score),
            "target_risk_score": str(target_score),
            "achieved_reduction": str(achieved_reduction),
            "required_reduction": str(required_reduction),
            "reduction_gap": str(gap),
            "verification_result": result_status,
            "measures_completed": len(measures),
            "total_measures": len([
                m for m in self._measures.values()
            ]),
            "verified_at": now.isoformat(),
            "provenance_hash": _compute_hash({
                "report_id": report_id,
                "strategy_id": strategy_id,
                "initial_score": str(initial_score),
                "current_score": str(current_score),
                "verified_at": now.isoformat(),
            }),
        }

        self._verification_reports[report_id] = verification_report

        # Record provenance
        if self._provenance is not None:
            self._provenance.record(
                entity_type="verification",
                action="verify",
                entity_id=report_id,
                actor="system",
                metadata={
                    "strategy_id": strategy_id,
                    "result": result_status,
                    "achieved_reduction": str(achieved_reduction),
                },
            )

        _safe_record(record_verification_run)
        elapsed = time.monotonic() - start
        _safe_observe(observe_verification_duration, elapsed)

        logger.info(
            f"Verification {report_id} for strategy {strategy_id}: "
            f"{result_status} (reduction={achieved_reduction}, "
            f"current={current_score}, target={target_score})"
        )

        return verification_report

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    async def generate_report(
        self,
        strategy_id: str,
    ) -> Dict[str, Any]:
        """Generate a mitigation report for a strategy.

        Produces a structured report suitable for inclusion in
        Due Diligence Statements (DDS) per EUDR Article 12(2),
        including measure summaries, effectiveness data, evidence
        references, and provenance hashes.

        Args:
            strategy_id: Strategy identifier.

        Returns:
            Mitigation report dictionary.

        Raises:
            ValueError: If strategy not found.
        """
        start = time.monotonic()

        logger.info(
            f"Generating mitigation report for strategy {strategy_id}"
        )

        # Delegate to engine if available
        if self._report_generator is not None:
            try:
                result = await self._report_generator.generate_report(
                    strategy_id=strategy_id,
                )
                _safe_record(record_report_generated)
                elapsed = time.monotonic() - start
                _safe_observe(observe_report_generation_duration, elapsed)
                return result
            except Exception as e:
                logger.debug(f"Engine generate_report failed: {e}")

        # Fallback: generate from in-memory data
        strategy = self._strategies.get(strategy_id)
        if strategy is None:
            raise ValueError(f"Strategy {strategy_id} not found")

        report_id = _new_uuid()
        now = _utcnow()

        # Collect measure summaries
        strategy_measures = strategy.get("measures", [])
        measure_summaries = []
        for m in strategy_measures:
            mid = m.get("measure_id", "")
            current_measure = self._measures.get(mid, m)
            evidence_items = self._evidence.get(mid, [])

            measure_summaries.append({
                "measure_id": mid,
                "title": current_measure.get("title", ""),
                "category": current_measure.get("category", ""),
                "status": current_measure.get("status", "proposed"),
                "estimated_reduction": current_measure.get(
                    "estimated_reduction", "0"
                ),
                "actual_reduction": current_measure.get(
                    "actual_reduction"
                ),
                "evidence_count": len(evidence_items),
            })

        # Get verification report if exists
        verifications = [
            v for v in self._verification_reports.values()
            if v.get("strategy_id") == strategy_id
        ]
        latest_verification = verifications[-1] if verifications else None

        report = {
            "report_id": report_id,
            "strategy_id": strategy_id,
            "operator_id": strategy.get("operator_id", ""),
            "commodity": strategy.get("commodity", ""),
            "risk_dimension": strategy.get("risk_dimension", ""),
            "current_risk_score": strategy.get("current_risk_score", ""),
            "target_risk_score": strategy.get("target_risk_score", ""),
            "status": strategy.get("status", ""),
            "measures": measure_summaries,
            "total_measures": len(measure_summaries),
            "completed_measures": sum(
                1 for m in measure_summaries
                if m["status"] == "completed"
            ),
            "verification": latest_verification,
            "generated_at": now.isoformat(),
            "regulation_reference": "EU 2023/1115 Article 11",
            "provenance_hash": _compute_hash({
                "report_id": report_id,
                "strategy_id": strategy_id,
                "generated_at": now.isoformat(),
                "measure_count": len(measure_summaries),
            }),
        }

        self._reports[report_id] = report

        # Record provenance
        if self._provenance is not None:
            self._provenance.record(
                entity_type="report",
                action="generate",
                entity_id=report_id,
                actor="system",
                metadata={
                    "strategy_id": strategy_id,
                    "measure_count": len(measure_summaries),
                },
            )

        _safe_record(record_report_generated)
        elapsed = time.monotonic() - start
        _safe_observe(observe_report_generation_duration, elapsed)

        logger.info(
            f"Mitigation report {report_id} generated for strategy "
            f"{strategy_id} in {elapsed * 1000:.1f}ms"
        )

        return report

    # ------------------------------------------------------------------
    # Workflows
    # ------------------------------------------------------------------

    async def initiate_workflow(
        self,
        risk_trigger: Any,
    ) -> Dict[str, Any]:
        """Start a complete mitigation workflow.

        Orchestrates the full mitigation lifecycle: trigger analysis,
        strategy design, measure creation, and workflow state tracking.

        Args:
            risk_trigger: RiskTrigger model with risk source details.

        Returns:
            Workflow state dictionary.
        """
        start = time.monotonic()

        logger.info(
            f"Initiating mitigation workflow for "
            f"operator={risk_trigger.operator_id}"
        )

        # Delegate to engine if available
        if self._compliance_workflow is not None:
            try:
                result = await self._compliance_workflow.initiate_workflow(
                    risk_trigger=risk_trigger,
                )
                _safe_record(record_workflow_initiated)
                _safe_gauge(set_active_workflows, len(self._workflows))
                return result
            except Exception as e:
                logger.debug(f"Engine initiate_workflow failed: {e}")

        # Fallback: create workflow and design strategy
        workflow_id = _new_uuid()
        now = _utcnow()

        # Design strategy
        strategy = await self.design_strategy(risk_trigger)
        strategy_id = (
            strategy.get("strategy_id")
            if isinstance(strategy, dict)
            else getattr(strategy, "strategy_id", _new_uuid())
        )

        # Update strategy with the link
        if isinstance(strategy, dict):
            strategy["strategy_id"] = strategy_id

        workflow = {
            "workflow_id": workflow_id,
            "strategy_id": strategy_id,
            "operator_id": risk_trigger.operator_id,
            "commodity": risk_trigger.commodity.value,
            "risk_level": risk_trigger.risk_level.value,
            "risk_dimension": risk_trigger.risk_dimension.value,
            "status": "initiated",
            "phase": "strategy_design",
            "created_at": now.isoformat(),
            "phases_completed": ["trigger_analysis", "strategy_design"],
            "phases_remaining": [
                "approval",
                "implementation",
                "verification",
                "reporting",
                "closure",
            ],
        }

        self._workflows[workflow_id] = workflow

        # Record provenance
        if self._provenance is not None:
            self._provenance.record(
                entity_type="workflow",
                action="initiate",
                entity_id=workflow_id,
                actor=risk_trigger.operator_id,
                metadata={
                    "strategy_id": strategy_id,
                    "commodity": risk_trigger.commodity.value,
                },
            )

        _safe_record(record_workflow_initiated)
        _safe_gauge(set_active_workflows, len(self._workflows))
        elapsed = time.monotonic() - start

        logger.info(
            f"Workflow {workflow_id} initiated in {elapsed * 1000:.1f}ms "
            f"(strategy={strategy_id})"
        )

        return workflow

    async def get_workflow_status(
        self,
        workflow_id: str,
    ) -> Dict[str, Any]:
        """Get the current status of a workflow.

        Args:
            workflow_id: Workflow identifier.

        Returns:
            Workflow state dictionary.

        Raises:
            ValueError: If workflow not found.
        """
        # Delegate to engine if available
        if self._compliance_workflow is not None:
            try:
                return await self._compliance_workflow.get_workflow_status(
                    workflow_id=workflow_id,
                )
            except Exception as e:
                logger.debug(f"Engine get_workflow_status failed: {e}")

        # Fallback: in-memory lookup
        workflow = self._workflows.get(workflow_id)
        if workflow is None:
            raise ValueError(f"Workflow {workflow_id} not found")

        return workflow

    # ------------------------------------------------------------------
    # Effectiveness Estimation
    # ------------------------------------------------------------------

    async def estimate_effectiveness(
        self,
        measure_id: str,
    ) -> Dict[str, Any]:
        """Estimate the effectiveness of a measure.

        Produces three-scenario (conservative/moderate/optimistic)
        projections using configured effectiveness factors.

        Args:
            measure_id: Measure identifier.

        Returns:
            Effectiveness estimate dictionary with three scenarios.

        Raises:
            ValueError: If measure not found.
        """
        start = time.monotonic()

        # Delegate to engine if available
        if self._effectiveness_estimator is not None:
            try:
                result = await self._effectiveness_estimator.estimate(
                    measure_id=measure_id,
                )
                elapsed = time.monotonic() - start
                _safe_observe(
                    observe_effectiveness_estimation_duration, elapsed
                )
                return result
            except Exception as e:
                logger.debug(f"Engine estimate failed: {e}")

        # Fallback: compute from in-memory data
        measure = self._measures.get(measure_id)
        if measure is None:
            raise ValueError(f"Measure {measure_id} not found")

        base_reduction = Decimal(
            measure.get("estimated_reduction", "10")
        )

        conservative = (
            base_reduction * self.config.conservative_factor
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        moderate = (
            base_reduction * self.config.moderate_factor
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        optimistic = (
            base_reduction * self.config.optimistic_factor
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # Apply cap
        cap = self.config.max_effectiveness_cap
        conservative = min(conservative, cap)
        moderate = min(moderate, cap)
        optimistic = min(optimistic, cap)

        # Apply minimum threshold
        threshold = self.config.min_effectiveness_threshold
        conservative = max(conservative, threshold)
        moderate = max(moderate, threshold)
        optimistic = max(optimistic, threshold)

        estimate = {
            "measure_id": measure_id,
            "base_reduction": str(base_reduction),
            "conservative": str(conservative),
            "moderate": str(moderate),
            "optimistic": str(optimistic),
            "factors": {
                "conservative_factor": str(self.config.conservative_factor),
                "moderate_factor": str(self.config.moderate_factor),
                "optimistic_factor": str(self.config.optimistic_factor),
            },
            "cap": str(cap),
            "threshold": str(threshold),
            "estimated_at": _utcnow().isoformat(),
        }

        elapsed = time.monotonic() - start
        _safe_observe(observe_effectiveness_estimation_duration, elapsed)

        return estimate

    # ------------------------------------------------------------------
    # Health and monitoring
    # ------------------------------------------------------------------

    async def health_check(self) -> Dict[str, Any]:
        """Perform a comprehensive health check.

        Checks database connectivity, Redis connectivity, engine
        availability, and returns detailed status information.

        Returns:
            Dictionary with health check results including overall
            status, engine statuses, and connection statuses.
        """
        result: Dict[str, Any] = {
            "agent_id": _AGENT_ID,
            "version": _VERSION,
            "status": "healthy",
            "initialized": self._initialized,
            "engines": {},
            "connections": {},
            "stores": {
                "strategies": len(self._strategies),
                "measures": len(self._measures),
                "workflows": len(self._workflows),
                "verification_reports": len(self._verification_reports),
                "reports": len(self._reports),
            },
            "timestamp": _utcnow().isoformat(),
        }

        # Check database
        db_status = "unavailable"
        if self._db_pool is not None:
            try:
                async with self._db_pool.connection() as conn:
                    await conn.execute("SELECT 1")
                db_status = "connected"
            except Exception:
                db_status = "error"
                result["status"] = "degraded"
        result["connections"]["postgresql"] = db_status

        # Check Redis
        redis_status = "unavailable"
        if self._redis is not None:
            try:
                await self._redis.ping()
                redis_status = "connected"
            except Exception:
                redis_status = "error"
        result["connections"]["redis"] = redis_status

        # Check engines
        expected_engines = [
            "mitigation_strategy_designer",
            "measure_template_library",
            "effectiveness_estimator",
            "measure_implementation_tracker",
            "risk_reduction_verifier",
            "compliance_workflow_engine",
            "mitigation_report_generator",
        ]

        for engine_name in expected_engines:
            if engine_name in self._engines:
                engine = self._engines[engine_name]
                if hasattr(engine, "health_check"):
                    try:
                        eng_health = await engine.health_check()
                        result["engines"][engine_name] = eng_health
                    except Exception as e:
                        result["engines"][engine_name] = {
                            "status": "error",
                            "error": str(e),
                        }
                else:
                    result["engines"][engine_name] = {
                        "status": "available"
                    }
            else:
                result["engines"][engine_name] = {
                    "status": "not_loaded"
                }

        # Determine overall status
        unhealthy_engines = sum(
            1
            for v in result["engines"].values()
            if isinstance(v, dict)
            and v.get("status") in ("error", "not_loaded")
        )
        if unhealthy_engines > 3:
            result["status"] = "unhealthy"
        elif unhealthy_engines > 0:
            result["status"] = "degraded"

        return result

    # ------------------------------------------------------------------
    # Component accessors
    # ------------------------------------------------------------------

    def get_engine(self, name: str) -> Optional[Any]:
        """Get a specific engine by name.

        Args:
            name: Engine name (e.g., 'mitigation_strategy_designer').

        Returns:
            Engine instance or None if not available.
        """
        return self._engines.get(name)

    @property
    def engine_count(self) -> int:
        """Return the number of loaded engines."""
        return len(self._engines)

    @property
    def is_initialized(self) -> bool:
        """Return whether the service has been initialized."""
        return self._initialized

    @property
    def strategy_count(self) -> int:
        """Return the number of strategies in memory."""
        return len(self._strategies)

    @property
    def measure_count(self) -> int:
        """Return the number of measures in memory."""
        return len(self._measures)

    @property
    def workflow_count(self) -> int:
        """Return the number of workflows in memory."""
        return len(self._workflows)


# ---------------------------------------------------------------------------
# Thread-safe singleton
# ---------------------------------------------------------------------------

_service_lock = threading.Lock()
_service_instance: Optional[MitigationMeasureDesignerService] = None


def get_service(
    config: Optional[MitigationMeasureDesignerConfig] = None,
) -> MitigationMeasureDesignerService:
    """Get the global MitigationMeasureDesignerService singleton instance.

    Thread-safe lazy initialization. Creates a new service instance
    on first call.

    Args:
        config: Optional configuration override for first creation.

    Returns:
        MitigationMeasureDesignerService singleton instance.

    Example:
        >>> service = get_service()
        >>> assert service is get_service()
    """
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = MitigationMeasureDesignerService(config)
    return _service_instance


def reset_service() -> None:
    """Reset the global service singleton to None.

    Used for testing teardown.
    """
    global _service_instance
    with _service_lock:
        _service_instance = None


# ---------------------------------------------------------------------------
# FastAPI lifespan context manager
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: Any) -> AsyncIterator[None]:
    """FastAPI lifespan context manager for startup/shutdown.

    Usage:
        >>> from fastapi import FastAPI
        >>> from greenlang.agents.eudr.mitigation_measure_designer.setup import lifespan
        >>> app = FastAPI(lifespan=lifespan)

    Args:
        app: FastAPI application instance.

    Yields:
        None -- application runs between startup and shutdown.
    """
    service = get_service()
    await service.startup()
    logger.info("Mitigation Measure Designer lifespan: startup complete")

    try:
        yield
    finally:
        await service.shutdown()
        logger.info(
            "Mitigation Measure Designer lifespan: shutdown complete"
        )
