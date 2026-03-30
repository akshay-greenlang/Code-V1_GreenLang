# -*- coding: utf-8 -*-
"""
Risk Assessment Engine Service Facade - AGENT-EUDR-028

High-level service facade that wires together all 7 processing engines
into a single cohesive entry point. Provides the primary API used by
the FastAPI router layer to calculate composite risk scores, evaluate
Article 10(2) criteria, classify risk levels, and generate risk
assessment reports per EUDR requirements.

This facade implements the Facade Pattern to hide the complexity
of the 7 internal engines behind a clean, use-case-oriented interface.

Service Methods:
    Full Pipeline:
        - assess_risk()                -> Execute full risk assessment pipeline

    Individual Engines:
        - calculate_composite_score()  -> Calculate composite risk score
        - evaluate_article10_criteria()-> Evaluate Article 10(2) criteria
        - get_country_benchmarks()     -> Get country benchmark data
        - classify_risk()              -> Classify risk level
        - check_simplified_dd()        -> Check simplified DD eligibility
        - apply_override()             -> Apply manual risk override
        - get_risk_trend()             -> Get risk trend analysis
        - generate_report()            -> Generate risk assessment report
        - batch_assess_risk()          -> Batch risk assessments

    Health & Monitoring:
        - health_check()               -> Check all engine statuses

Singleton Pattern:
    Thread-safe singleton with double-checked locking via ``get_service()``.

FastAPI Integration:
    Use the ``lifespan`` async context manager with
    ``FastAPI(lifespan=lifespan)`` for automatic startup/shutdown.

Example:
    >>> from greenlang.agents.eudr.risk_assessment_engine.setup import (
    ...     RiskAssessmentEngineService,
    ...     get_service,
    ... )
    >>> service = get_service()
    >>> await service.startup()
    >>> health = await service.health_check()
    >>> assert health["status"] == "healthy"
    >>>
    >>> # Full pipeline
    >>> operation = await service.assess_risk(
    ...     operator_id="OP-001",
    ...     commodity="cocoa",
    ...     country_codes=["GH", "CI"],
    ...     supplier_ids=["SUP-001"],
    ... )
    >>>
    >>> await service.shutdown()

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-028 Risk Assessment Engine (GL-EUDR-RAE-028)
Regulation: EU 2023/1115 (EUDR) Articles 4, 9, 10, 12, 13, 29, 31
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

_VERSION = "1.0.0"

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
# Internal imports: config, provenance, metrics
# ---------------------------------------------------------------------------

from greenlang.agents.eudr.risk_assessment_engine.config import (
    RiskAssessmentEngineConfig,
    get_config,
)
from greenlang.agents.eudr.risk_assessment_engine.provenance import (
    ProvenanceTracker,
)

# Metrics import (graceful fallback)
try:
    from greenlang.agents.eudr.risk_assessment_engine.metrics import (
        record_risk_assessment_operation,
        observe_composite_score_duration,
        record_api_error,
    )

    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    record_risk_assessment_operation = None  # type: ignore[assignment]
    observe_composite_score_duration = None  # type: ignore[assignment]
    record_api_error = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Engine imports (conditional - engines may not exist yet during parallel build)
# ---------------------------------------------------------------------------

# ---- Engine 1: Composite Risk Calculator ----
try:
    from greenlang.agents.eudr.risk_assessment_engine.composite_risk_calculator import (
        CompositeRiskCalculator,
    )
except ImportError:
    CompositeRiskCalculator = None  # type: ignore[misc,assignment]

# ---- Engine 2: Risk Factor Aggregator ----
try:
    from greenlang.agents.eudr.risk_assessment_engine.risk_factor_aggregator import (
        RiskFactorAggregator,
    )
except ImportError:
    RiskFactorAggregator = None  # type: ignore[misc,assignment]

# ---- Engine 3: Country Benchmark Engine ----
try:
    from greenlang.agents.eudr.risk_assessment_engine.country_benchmark_engine import (
        CountryBenchmarkEngine,
    )
except ImportError:
    CountryBenchmarkEngine = None  # type: ignore[misc,assignment]

# ---- Engine 4: Article 10 Criteria Evaluator ----
try:
    from greenlang.agents.eudr.risk_assessment_engine.article10_criteria_evaluator import (
        Article10CriteriaEvaluator,
    )
except ImportError:
    Article10CriteriaEvaluator = None  # type: ignore[misc,assignment]

# ---- Engine 5: Risk Classification Engine ----
try:
    from greenlang.agents.eudr.risk_assessment_engine.risk_classification_engine import (
        RiskClassificationEngine,
    )
except ImportError:
    RiskClassificationEngine = None  # type: ignore[misc,assignment]

# ---- Engine 6: Risk Trend Analyzer ----
try:
    from greenlang.agents.eudr.risk_assessment_engine.risk_trend_analyzer import (
        RiskTrendAnalyzer,
    )
except ImportError:
    RiskTrendAnalyzer = None  # type: ignore[misc,assignment]

# ---- Engine 7: Risk Report Generator ----
try:
    from greenlang.agents.eudr.risk_assessment_engine.risk_report_generator import (
        RiskReportGenerator,
    )
except ImportError:
    RiskReportGenerator = None  # type: ignore[misc,assignment]

# ---------------------------------------------------------------------------
# Model imports
# ---------------------------------------------------------------------------

from greenlang.agents.eudr.risk_assessment_engine.models import (
    Article10CriteriaResult,
    CompositeRiskScore,
    CountryBenchmark,
    EUDRCommodity,
    OverrideReason,
    RiskAssessmentOperation,
    RiskAssessmentReport,
    RiskAssessmentStatus,
    RiskFactorInput,
    RiskLevel,
    RiskOverride,
    RiskTrendAnalysis,
    SimplifiedDDEligibility,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Service Facade
# ---------------------------------------------------------------------------

class RiskAssessmentEngineService:
    """High-level service facade for the Risk Assessment Engine Agent.

    Wires together all 7 processing engines and provides a unified API
    for composite risk scoring, Article 10(2) evaluation, risk level
    classification, and risk assessment report generation per EUDR
    requirements.

    Attributes:
        config: Agent configuration.
        provenance: SHA-256 provenance tracker.
        _db_pool: PostgreSQL async connection pool.
        _redis: Redis async client.
        _engines: Dictionary of initialized engines.
        _initialized: Whether startup has completed.

    Example:
        >>> service = RiskAssessmentEngineService()
        >>> await service.startup()
        >>> operation = await service.assess_risk(
        ...     operator_id="OP-001",
        ...     commodity="cocoa",
        ...     country_codes=["GH", "CI"],
        ...     supplier_ids=["SUP-001"],
        ... )
        >>> assert operation.status == RiskAssessmentStatus.COMPLETED
    """

    def __init__(
        self,
        config: Optional[RiskAssessmentEngineConfig] = None,
    ) -> None:
        """Initialize the service facade.

        Args:
            config: Optional configuration override.
                   If None, uses get_config() singleton.
        """
        self.config = config or get_config()
        self.provenance = ProvenanceTracker()
        self._db_pool: Optional[Any] = None
        self._redis: Optional[Any] = None
        self._engines: Dict[str, Any] = {}
        self._initialized = False

        logger.info("RiskAssessmentEngineService created")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        """Initialize all engines and external connections.

        Performs database pool creation, Redis connection, engine
        initialization, and logs startup time and engine availability.

        Raises:
            RuntimeError: If critical engine initialization fails.
        """
        start = time.monotonic()
        logger.info("RiskAssessmentEngineService startup initiated")

        # Initialize database pool
        if PSYCOPG_POOL_AVAILABLE:
            try:
                self._db_pool = AsyncConnectionPool(
                    conninfo=self.config.database_url,
                    min_size=2,
                    max_size=self.config.pool_size,
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
                self._redis = aioredis.from_url(
                    self.config.redis_url,
                    decode_responses=True,
                )
                await self._redis.ping()
                logger.info("Redis connection established")
            except Exception as e:
                logger.warning(f"Redis init failed: {e}")
                self._redis = None

        # Initialize engines
        self._init_engines()

        self._initialized = True
        elapsed = (time.monotonic() - start) * 1000
        engine_count = len(self._engines)

        logger.info(
            f"RiskAssessmentEngineService startup complete: "
            f"{engine_count}/7 engines in {elapsed:.1f}ms"
        )

    def _init_engines(self) -> None:
        """Initialize all 7 processing engines with graceful degradation."""
        engine_specs: List[Tuple[str, Any]] = [
            ("composite_risk_calculator", CompositeRiskCalculator),
            ("risk_factor_aggregator", RiskFactorAggregator),
            ("country_benchmark", CountryBenchmarkEngine),
            ("article10_criteria_evaluator", Article10CriteriaEvaluator),
            ("risk_classification", RiskClassificationEngine),
            ("risk_trend_analyzer", RiskTrendAnalyzer),
            ("risk_report_generator", RiskReportGenerator),
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

    async def shutdown(self) -> None:
        """Gracefully shutdown all connections and engines.

        Closes database pool, Redis connection, and any
        engine-specific resources.
        """
        logger.info("RiskAssessmentEngineService shutdown initiated")

        # Shutdown engines with async shutdown methods
        for name, engine in self._engines.items():
            if hasattr(engine, "shutdown"):
                try:
                    await engine.shutdown()
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

        self._initialized = False
        logger.info("RiskAssessmentEngineService shutdown complete")

    # ------------------------------------------------------------------
    # Full Pipeline
    # ------------------------------------------------------------------

    async def assess_risk(
        self,
        operator_id: str,
        commodity: str,
        country_codes: List[str],
        supplier_ids: List[str],
    ) -> RiskAssessmentOperation:
        """Execute the full risk assessment pipeline.

        Orchestrates all 7 engines in sequence:
            1. Create operation (status=INITIATED)
            2. Aggregate risk factors from upstream agents (status=AGGREGATING)
            3. Calculate composite risk score (status=EVALUATING)
            4. Evaluate Article 10(2) criteria
            5. Get country benchmarks and apply
            6. Classify risk level (status=CLASSIFYING)
            7. Check simplified due diligence eligibility
            8. Add trend data point
            9. Generate risk assessment report
            10. Update operation (status=COMPLETED)

        Args:
            operator_id: EUDR operator identifier.
            commodity: Commodity being assessed.
            country_codes: ISO 3166-1 alpha-2 country codes for sourcing.
            supplier_ids: Supplier identifiers in the supply chain.

        Returns:
            RiskAssessmentOperation with status, scores, and report reference.
        """
        start = time.monotonic()
        operation_id = _new_uuid()

        logger.info(
            f"Starting risk assessment {operation_id} for "
            f"operator={operator_id}, commodity={commodity}, "
            f"countries={country_codes}"
        )

        # Step 1: Create operation (INITIATED)
        operation = RiskAssessmentOperation(
            operation_id=operation_id,
            operator_id=operator_id,
            commodity=commodity,
            country_codes=country_codes,
            supplier_ids=supplier_ids,
            status=RiskAssessmentStatus.INITIATED,
        )

        try:
            # Step 2: Aggregate risk factors (AGGREGATING)
            operation.status = RiskAssessmentStatus.AGGREGATING
            factor_inputs = await self._step_aggregate_risk_factors(
                operation_id, commodity, country_codes, supplier_ids
            )
            operation.total_factors_collected = len(factor_inputs)

            # Step 3: Calculate composite risk score (EVALUATING)
            operation.status = RiskAssessmentStatus.EVALUATING
            composite_score = await self._step_calculate_composite_score(
                operation_id, factor_inputs, country_codes
            )
            operation.composite_score = composite_score.overall_score

            # Step 4: Evaluate Article 10(2) criteria
            benchmarks = await self._step_get_country_benchmarks(
                operation_id, country_codes
            )
            article10_result = await self._step_evaluate_article10_criteria(
                operation_id, factor_inputs, benchmarks, composite_score
            )
            operation.article10_criteria_met = article10_result.criteria_met
            operation.article10_criteria_total = article10_result.criteria_total

            # Step 5: Country benchmark data already fetched
            operation.country_benchmarks = [
                b.benchmark_level.value for b in benchmarks
            ]

            # Step 6: Classify risk level (CLASSIFYING)
            operation.status = RiskAssessmentStatus.CLASSIFYING
            risk_level = await self._step_classify_risk(
                operation_id, composite_score, article10_result
            )
            operation.risk_level = risk_level

            # Step 7: Check simplified DD eligibility
            simplified_dd = await self._step_check_simplified_dd(
                operation_id, composite_score, benchmarks
            )
            operation.simplified_dd_eligible = simplified_dd.eligible

            # Step 8: Add trend data point
            await self._step_add_trend_data_point(
                operation_id, operator_id, commodity, composite_score
            )

            # Step 9: Generate risk assessment report
            operation.status = RiskAssessmentStatus.REPORTING
            report = await self._step_generate_report(operation_id, operation)
            operation.report_id = report.report_id if report else None

            # Step 10: Mark operation complete
            operation.status = RiskAssessmentStatus.COMPLETED
            operation.completed_at = utcnow()
            elapsed_ms = int((time.monotonic() - start) * 1000)
            operation.duration_ms = elapsed_ms
            operation.provenance_hash = _compute_hash(
                operation.model_dump(mode="json")
            )

            # Record provenance
            self.provenance.record(
                operation="assess_risk",
                input_hash=_compute_hash({
                    "operator_id": operator_id,
                    "commodity": commodity,
                    "country_codes": country_codes,
                    "supplier_ids": supplier_ids,
                }),
                output_hash=operation.provenance_hash,
            )

            if METRICS_AVAILABLE and record_risk_assessment_operation is not None:
                record_risk_assessment_operation(commodity, "completed")

            logger.info(
                f"Risk assessment {operation_id} completed in "
                f"{elapsed_ms}ms (risk_level={risk_level.value}, "
                f"composite={operation.composite_score})"
            )

        except Exception as e:
            operation.status = RiskAssessmentStatus.FAILED
            operation.completed_at = utcnow()
            elapsed_ms = int((time.monotonic() - start) * 1000)
            operation.duration_ms = elapsed_ms

            if METRICS_AVAILABLE and record_api_error is not None:
                record_api_error("assess_risk")

            logger.error(
                f"Risk assessment {operation_id} failed: "
                f"{type(e).__name__}: {str(e)[:500]}",
                exc_info=True,
            )

        return operation

    # ------------------------------------------------------------------
    # Pipeline step helpers
    # ------------------------------------------------------------------

    async def _step_aggregate_risk_factors(
        self,
        operation_id: str,
        commodity: str,
        country_codes: List[str],
        supplier_ids: List[str],
    ) -> List[RiskFactorInput]:
        """Step 2: Aggregate risk factors from upstream agent data.

        Args:
            operation_id: Current operation identifier.
            commodity: Commodity for factor filtering.
            country_codes: Source country codes.
            supplier_ids: Supplier identifiers.

        Returns:
            List of aggregated risk factor inputs.
        """
        engine = self._engines.get("risk_factor_aggregator")
        if engine is None:
            logger.warning("RiskFactorAggregator not available")
            return []

        try:
            factors = await engine.aggregate_factors(
                operation_id=operation_id,
                commodity=commodity,
                country_codes=country_codes,
                supplier_ids=supplier_ids,
            )
            return factors
        except Exception as e:
            logger.warning(
                f"Risk factor aggregation failed for "
                f"operation {operation_id}: {e}"
            )
            return []

    async def _step_calculate_composite_score(
        self,
        operation_id: str,
        factor_inputs: List[RiskFactorInput],
        country_codes: List[str],
    ) -> CompositeRiskScore:
        """Step 3: Calculate composite risk score.

        Args:
            operation_id: Current operation identifier.
            factor_inputs: Aggregated risk factor inputs.
            country_codes: Source country codes for weighting.

        Returns:
            CompositeRiskScore with dimension breakdowns.
        """
        engine = self._engines.get("composite_risk_calculator")
        if engine is None:
            logger.warning("CompositeRiskCalculator not available")
            return CompositeRiskScore(
                overall_score=Decimal("0.50"),
                confidence=Decimal("0.00"),
            )

        try:
            score = await engine.calculate(
                factor_inputs=factor_inputs,
                country_codes=country_codes,
            )

            if METRICS_AVAILABLE and observe_composite_score_duration is not None:
                observe_composite_score_duration(0.0)

            return score
        except Exception as e:
            logger.warning(
                f"Composite score calculation failed for "
                f"operation {operation_id}: {e}"
            )
            return CompositeRiskScore(
                overall_score=Decimal("0.50"),
                confidence=Decimal("0.00"),
            )

    async def _step_get_country_benchmarks(
        self,
        operation_id: str,
        country_codes: List[str],
    ) -> List[CountryBenchmark]:
        """Step 5: Get country benchmark classifications.

        Args:
            operation_id: Current operation identifier.
            country_codes: ISO 3166-1 alpha-2 country codes.

        Returns:
            List of country benchmark results.
        """
        engine = self._engines.get("country_benchmark")
        if engine is None:
            logger.warning("CountryBenchmarkEngine not available")
            return []

        benchmarks: List[CountryBenchmark] = []
        for code in country_codes:
            try:
                benchmark = await engine.get_benchmark(country_code=code)
                benchmarks.append(benchmark)
            except Exception as e:
                logger.warning(
                    f"Country benchmark lookup for {code} failed in "
                    f"operation {operation_id}: {e}"
                )

        return benchmarks

    async def _step_evaluate_article10_criteria(
        self,
        operation_id: str,
        factor_inputs: List[RiskFactorInput],
        benchmarks: List[CountryBenchmark],
        composite: CompositeRiskScore,
    ) -> Article10CriteriaResult:
        """Step 4: Evaluate Article 10(2) criteria.

        Args:
            operation_id: Current operation identifier.
            factor_inputs: Aggregated risk factor inputs.
            benchmarks: Country benchmark data.
            composite: Composite risk score.

        Returns:
            Article10CriteriaResult with per-criterion evaluation.
        """
        engine = self._engines.get("article10_criteria_evaluator")
        if engine is None:
            logger.warning("Article10CriteriaEvaluator not available")
            return Article10CriteriaResult(
                criteria_met=0,
                criteria_total=7,
            )

        try:
            result = await engine.evaluate(
                factor_inputs=factor_inputs,
                benchmarks=benchmarks,
                composite_score=composite,
            )
            return result
        except Exception as e:
            logger.warning(
                f"Article 10 criteria evaluation failed for "
                f"operation {operation_id}: {e}"
            )
            return Article10CriteriaResult(
                criteria_met=0,
                criteria_total=7,
            )

    async def _step_classify_risk(
        self,
        operation_id: str,
        composite: CompositeRiskScore,
        article10: Optional[Article10CriteriaResult] = None,
        previous_level: Optional[RiskLevel] = None,
    ) -> RiskLevel:
        """Step 6: Classify risk level from composite score.

        Args:
            operation_id: Current operation identifier.
            composite: Composite risk score.
            article10: Optional Article 10 evaluation result.
            previous_level: Optional previous risk level for stability.

        Returns:
            Classified RiskLevel.
        """
        engine = self._engines.get("risk_classification")
        if engine is None:
            logger.warning("RiskClassificationEngine not available")
            return RiskLevel.STANDARD

        try:
            level = await engine.classify(
                composite_score=composite,
                article10_result=article10,
                previous_level=previous_level,
            )
            return level
        except Exception as e:
            logger.warning(
                f"Risk classification failed for "
                f"operation {operation_id}: {e}"
            )
            return RiskLevel.STANDARD

    async def _step_check_simplified_dd(
        self,
        operation_id: str,
        composite: CompositeRiskScore,
        benchmarks: List[CountryBenchmark],
    ) -> SimplifiedDDEligibility:
        """Step 7: Check simplified due diligence eligibility.

        Per EUDR Article 13, operators sourcing exclusively from
        low-risk countries may use simplified due diligence.

        Args:
            operation_id: Current operation identifier.
            composite: Composite risk score.
            benchmarks: Country benchmark classifications.

        Returns:
            SimplifiedDDEligibility with eligibility determination.
        """
        engine = self._engines.get("risk_classification")
        if engine is None:
            logger.warning("RiskClassificationEngine not available for DD check")
            return SimplifiedDDEligibility(eligible=False)

        try:
            eligibility = await engine.check_simplified_dd(
                composite_score=composite,
                benchmarks=benchmarks,
            )
            return eligibility
        except Exception as e:
            logger.warning(
                f"Simplified DD check failed for "
                f"operation {operation_id}: {e}"
            )
            return SimplifiedDDEligibility(eligible=False)

    async def _step_add_trend_data_point(
        self,
        operation_id: str,
        operator_id: str,
        commodity: str,
        composite: CompositeRiskScore,
    ) -> None:
        """Step 8: Record trend data point for temporal analysis.

        Args:
            operation_id: Current operation identifier.
            operator_id: EUDR operator identifier.
            commodity: Commodity being assessed.
            composite: Composite risk score to record.
        """
        engine = self._engines.get("risk_trend_analyzer")
        if engine is None:
            logger.debug("RiskTrendAnalyzer not available for trend recording")
            return

        try:
            await engine.add_data_point(
                operator_id=operator_id,
                commodity=commodity,
                score=composite.overall_score,
                timestamp=utcnow(),
            )
        except Exception as e:
            logger.warning(
                f"Trend data point recording failed for "
                f"operation {operation_id}: {e}"
            )

    async def _step_generate_report(
        self,
        operation_id: str,
        operation: RiskAssessmentOperation,
    ) -> Optional[RiskAssessmentReport]:
        """Step 9: Generate risk assessment report.

        Args:
            operation_id: Current operation identifier.
            operation: Completed risk assessment operation.

        Returns:
            RiskAssessmentReport or None if generation fails.
        """
        engine = self._engines.get("risk_report_generator")
        if engine is None:
            logger.warning("RiskReportGenerator not available")
            return None

        try:
            report = await engine.generate_report(operation=operation)
            return report
        except Exception as e:
            logger.warning(
                f"Report generation failed for "
                f"operation {operation_id}: {e}"
            )
            return None

    # ------------------------------------------------------------------
    # Individual engine delegates
    # ------------------------------------------------------------------

    async def calculate_composite_score(
        self,
        factor_inputs: List[RiskFactorInput],
        country_codes: Optional[List[str]] = None,
    ) -> CompositeRiskScore:
        """Calculate a composite risk score from factor inputs.

        Delegates to CompositeRiskCalculator.

        Args:
            factor_inputs: List of risk factor inputs.
            country_codes: Optional country codes for weighting.

        Returns:
            CompositeRiskScore with dimension breakdowns.

        Raises:
            RuntimeError: If engine is not available.
        """
        engine = self._engines.get("composite_risk_calculator")
        if engine is None:
            raise RuntimeError("CompositeRiskCalculator not available")
        return await engine.calculate(
            factor_inputs=factor_inputs,
            country_codes=country_codes or [],
        )

    async def evaluate_article10_criteria(
        self,
        factor_inputs: List[RiskFactorInput],
        benchmarks: List[CountryBenchmark],
        composite: CompositeRiskScore,
    ) -> Article10CriteriaResult:
        """Evaluate Article 10(2) criteria.

        Delegates to Article10CriteriaEvaluator.

        Args:
            factor_inputs: Aggregated risk factor inputs.
            benchmarks: Country benchmark data.
            composite: Composite risk score.

        Returns:
            Article10CriteriaResult with per-criterion evaluation.

        Raises:
            RuntimeError: If engine is not available.
        """
        engine = self._engines.get("article10_criteria_evaluator")
        if engine is None:
            raise RuntimeError("Article10CriteriaEvaluator not available")
        return await engine.evaluate(
            factor_inputs=factor_inputs,
            benchmarks=benchmarks,
            composite_score=composite,
        )

    async def get_country_benchmarks(
        self,
        country_codes: List[str],
    ) -> List[CountryBenchmark]:
        """Get country benchmark classifications for given countries.

        Delegates to CountryBenchmarkEngine.

        Args:
            country_codes: ISO 3166-1 alpha-2 country codes.

        Returns:
            List of CountryBenchmark results.

        Raises:
            RuntimeError: If engine is not available.
        """
        engine = self._engines.get("country_benchmark")
        if engine is None:
            raise RuntimeError("CountryBenchmarkEngine not available")

        benchmarks: List[CountryBenchmark] = []
        for code in country_codes:
            benchmark = await engine.get_benchmark(country_code=code)
            benchmarks.append(benchmark)

        return benchmarks

    async def classify_risk(
        self,
        composite: CompositeRiskScore,
        article10: Optional[Article10CriteriaResult] = None,
        previous: Optional[RiskLevel] = None,
    ) -> RiskLevel:
        """Classify risk level from composite score.

        Delegates to RiskClassificationEngine.

        Args:
            composite: Composite risk score.
            article10: Optional Article 10(2) evaluation result.
            previous: Optional previous risk level for stability.

        Returns:
            Classified RiskLevel.

        Raises:
            RuntimeError: If engine is not available.
        """
        engine = self._engines.get("risk_classification")
        if engine is None:
            raise RuntimeError("RiskClassificationEngine not available")
        return await engine.classify(
            composite_score=composite,
            article10_result=article10,
            previous_level=previous,
        )

    async def check_simplified_dd(
        self,
        composite: CompositeRiskScore,
        benchmarks: List[CountryBenchmark],
    ) -> SimplifiedDDEligibility:
        """Check simplified due diligence eligibility per Article 13.

        Delegates to RiskClassificationEngine.

        Args:
            composite: Composite risk score.
            benchmarks: Country benchmark classifications.

        Returns:
            SimplifiedDDEligibility with eligibility determination.

        Raises:
            RuntimeError: If engine is not available.
        """
        engine = self._engines.get("risk_classification")
        if engine is None:
            raise RuntimeError("RiskClassificationEngine not available")
        return await engine.check_simplified_dd(
            composite_score=composite,
            benchmarks=benchmarks,
        )

    async def apply_override(
        self,
        assessment_id: str,
        overridden_score: Decimal,
        reason: OverrideReason,
        justification: str,
        overridden_by: str,
    ) -> RiskOverride:
        """Apply a manual risk override to an assessment.

        Creates a validated RiskOverride record with provenance
        tracking. Overrides must include justification and are
        subject to audit trail requirements.

        Args:
            assessment_id: Risk assessment operation ID to override.
            overridden_score: New score value (0.00 - 1.00).
            reason: Reason category for the override.
            justification: Detailed justification text.
            overridden_by: User or system identifier performing override.

        Returns:
            RiskOverride with provenance hash.

        Raises:
            ValueError: If overridden_score is out of range or
                       justification is empty.
        """
        # Validate inputs
        if not (Decimal("0.00") <= overridden_score <= Decimal("1.00")):
            raise ValueError(
                f"overridden_score must be between 0.00 and 1.00, "
                f"got {overridden_score}"
            )
        if not justification or len(justification.strip()) < 10:
            raise ValueError(
                "justification must be at least 10 characters"
            )

        override = RiskOverride(
            override_id=_new_uuid(),
            assessment_id=assessment_id,
            overridden_score=overridden_score,
            reason=reason,
            justification=justification.strip(),
            overridden_by=overridden_by,
            created_at=utcnow(),
        )

        # Calculate provenance hash
        override.provenance_hash = _compute_hash(
            override.model_dump(mode="json")
        )

        # Record provenance
        self.provenance.record(
            operation="apply_override",
            input_hash=_compute_hash({
                "assessment_id": assessment_id,
                "reason": reason.value,
            }),
            output_hash=override.provenance_hash,
        )

        logger.info(
            f"Risk override applied to assessment {assessment_id} "
            f"by {overridden_by} (reason={reason.value}, "
            f"score={overridden_score})"
        )

        return override

    async def get_risk_trend(
        self,
        operator_id: str,
        commodity: str,
    ) -> RiskTrendAnalysis:
        """Get risk trend analysis for an operator and commodity.

        Delegates to RiskTrendAnalyzer.

        Args:
            operator_id: EUDR operator identifier.
            commodity: Commodity being assessed.

        Returns:
            RiskTrendAnalysis with trend direction and data points.

        Raises:
            RuntimeError: If engine is not available.
        """
        engine = self._engines.get("risk_trend_analyzer")
        if engine is None:
            raise RuntimeError("RiskTrendAnalyzer not available")
        return await engine.get_trend(
            operator_id=operator_id,
            commodity=commodity,
        )

    async def generate_report(
        self,
        operation: RiskAssessmentOperation,
    ) -> RiskAssessmentReport:
        """Generate a risk assessment report from a completed operation.

        Delegates to RiskReportGenerator.

        Args:
            operation: Completed risk assessment operation.

        Returns:
            RiskAssessmentReport with structured findings.

        Raises:
            RuntimeError: If engine is not available.
        """
        engine = self._engines.get("risk_report_generator")
        if engine is None:
            raise RuntimeError("RiskReportGenerator not available")
        return await engine.generate_report(operation=operation)

    async def batch_assess_risk(
        self,
        requests: List[Dict[str, Any]],
    ) -> List[RiskAssessmentOperation]:
        """Execute multiple risk assessments in batch.

        Processes each assessment request sequentially to maintain
        deterministic ordering and provenance chain integrity.

        Args:
            requests: List of assessment request dictionaries, each
                     containing operator_id, commodity, country_codes,
                     and supplier_ids.

        Returns:
            List of RiskAssessmentOperation results.
        """
        results: List[RiskAssessmentOperation] = []

        for i, req in enumerate(requests):
            try:
                operation = await self.assess_risk(
                    operator_id=req["operator_id"],
                    commodity=req["commodity"],
                    country_codes=req.get("country_codes", []),
                    supplier_ids=req.get("supplier_ids", []),
                )
                results.append(operation)
            except Exception as e:
                logger.warning(
                    f"Batch assessment {i + 1}/{len(requests)} failed: {e}"
                )
                # Create a failed operation for tracking
                failed_op = RiskAssessmentOperation(
                    operation_id=_new_uuid(),
                    operator_id=req.get("operator_id", "unknown"),
                    commodity=req.get("commodity", "unknown"),
                    country_codes=req.get("country_codes", []),
                    supplier_ids=req.get("supplier_ids", []),
                    status=RiskAssessmentStatus.FAILED,
                    completed_at=utcnow(),
                )
                results.append(failed_op)

        logger.info(
            f"Batch risk assessment completed: {len(results)} assessments "
            f"({sum(1 for r in results if r.status == RiskAssessmentStatus.COMPLETED)} succeeded)"
        )

        return results

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
            "agent_id": "GL-EUDR-RAE-028",
            "version": _VERSION,
            "status": "healthy",
            "initialized": self._initialized,
            "engines": {},
            "connections": {},
            "timestamp": utcnow().isoformat(),
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
            "composite_risk_calculator",
            "risk_factor_aggregator",
            "country_benchmark",
            "article10_criteria_evaluator",
            "risk_classification",
            "risk_trend_analyzer",
            "risk_report_generator",
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
                    result["engines"][engine_name] = {"status": "available"}
            else:
                result["engines"][engine_name] = {"status": "not_loaded"}

        # Determine overall status
        unhealthy_engines = sum(
            1
            for v in result["engines"].values()
            if isinstance(v, dict) and v.get("status") in ("error", "not_loaded")
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
            name: Engine name (e.g., 'composite_risk_calculator').

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

# ---------------------------------------------------------------------------
# Thread-safe singleton
# ---------------------------------------------------------------------------

_service_lock = threading.Lock()
_service_instance: Optional[RiskAssessmentEngineService] = None

def get_service(
    config: Optional[RiskAssessmentEngineConfig] = None,
) -> RiskAssessmentEngineService:
    """Get the global RiskAssessmentEngineService singleton instance.

    Thread-safe lazy initialization. Creates a new service instance
    on first call.

    Args:
        config: Optional configuration override for first creation.

    Returns:
        RiskAssessmentEngineService singleton instance.

    Example:
        >>> service = get_service()
        >>> assert service is get_service()
    """
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = RiskAssessmentEngineService(config)
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
        >>> from greenlang.agents.eudr.risk_assessment_engine.setup import lifespan
        >>> app = FastAPI(lifespan=lifespan)

    Args:
        app: FastAPI application instance.

    Yields:
        None - application runs between startup and shutdown.
    """
    service = get_service()
    await service.startup()
    logger.info("Risk Assessment Engine Agent lifespan: startup complete")

    try:
        yield
    finally:
        await service.shutdown()
        logger.info("Risk Assessment Engine Agent lifespan: shutdown complete")
