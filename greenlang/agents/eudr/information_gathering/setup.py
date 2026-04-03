# -*- coding: utf-8 -*-
"""
Information Gathering Service Facade - AGENT-EUDR-027

High-level service facade that wires together all 7 processing engines
into a single cohesive entry point. Provides the primary API used by
the FastAPI router layer to collect, verify, normalize, and package
information required by EUDR Article 9 for due diligence statements.

This facade implements the Facade Pattern to hide the complexity
of the 7 internal engines behind a clean, use-case-oriented interface.

Service Methods:
    Full Pipeline:
        - gather_information()   -> Execute full information gathering pipeline

    Individual Engines:
        - query_external_database()     -> Query a single external database
        - verify_certificate()          -> Verify a single certificate
        - batch_verify_certificates()   -> Verify certificates in batch
        - harvest_public_data()         -> Harvest public data from source
        - aggregate_supplier()          -> Aggregate supplier information
        - validate_completeness()       -> Validate Article 9 completeness
        - assemble_package()            -> Assemble information package

    Health & Monitoring:
        - health_check()                -> Check all engine statuses

Singleton Pattern:
    Thread-safe singleton with double-checked locking via ``get_service()``.

FastAPI Integration:
    Use the ``lifespan`` async context manager with
    ``FastAPI(lifespan=lifespan)`` for automatic startup/shutdown.

Example:
    >>> from greenlang.agents.eudr.information_gathering.setup import (
    ...     InformationGatheringService,
    ...     get_service,
    ... )
    >>> service = get_service()
    >>> await service.startup()
    >>> health = await service.health_check()
    >>> assert health["status"] == "healthy"
    >>>
    >>> # Full pipeline
    >>> operation = await service.gather_information("OP-001", EUDRCommodity.COCOA)
    >>>
    >>> # Query a single external database
    >>> result = await service.query_external_database(
    ...     ExternalDatabaseSource.EU_TRACES, {"certificate_number": "TRACES-001"}
    ... )
    >>>
    >>> await service.shutdown()

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-027 Information Gathering Agent (GL-EUDR-IGA-027)
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

from greenlang.agents.eudr.information_gathering.config import (
    InformationGatheringConfig,
    get_config,
)
from greenlang.agents.eudr.information_gathering.provenance import (
    ProvenanceTracker,
)

# Metrics import (graceful fallback)
try:
    from greenlang.agents.eudr.information_gathering.metrics import (
        record_gathering_operation,
        observe_package_assembly_duration,
        record_api_error,
    )

    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    record_gathering_operation = None  # type: ignore[assignment]
    observe_package_assembly_duration = None  # type: ignore[assignment]
    record_api_error = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Engine imports (conditional - engines may not exist yet during parallel build)
# ---------------------------------------------------------------------------

# ---- Engine 1: External Database Connector ----
try:
    from greenlang.agents.eudr.information_gathering.external_database_connector import (
        ExternalDatabaseConnectorEngine,
    )
except ImportError:
    ExternalDatabaseConnectorEngine = None  # type: ignore[misc,assignment]

# ---- Engine 2: Certification Verification ----
try:
    from greenlang.agents.eudr.information_gathering.certification_verification_engine import (
        CertificationVerificationEngine,
    )
except ImportError:
    CertificationVerificationEngine = None  # type: ignore[misc,assignment]

# ---- Engine 3: Public Data Mining ----
try:
    from greenlang.agents.eudr.information_gathering.public_data_mining_engine import (
        PublicDataMiningEngine,
    )
except ImportError:
    PublicDataMiningEngine = None  # type: ignore[misc,assignment]

# ---- Engine 4: Supplier Information Aggregator ----
try:
    from greenlang.agents.eudr.information_gathering.supplier_information_aggregator import (
        SupplierInformationAggregator,
    )
except ImportError:
    SupplierInformationAggregator = None  # type: ignore[misc,assignment]

# ---- Engine 5: Completeness Validation ----
try:
    from greenlang.agents.eudr.information_gathering.completeness_validation_engine import (
        CompletenessValidationEngine,
    )
except ImportError:
    CompletenessValidationEngine = None  # type: ignore[misc,assignment]

# ---- Engine 6: Data Normalization ----
try:
    from greenlang.agents.eudr.information_gathering.data_normalization_engine import (
        DataNormalizationEngine,
    )
except ImportError:
    DataNormalizationEngine = None  # type: ignore[misc,assignment]

# ---- Engine 7: Package Assembly ----
try:
    from greenlang.agents.eudr.information_gathering.package_assembly_engine import (
        PackageAssemblyEngine,
    )
except ImportError:
    PackageAssemblyEngine = None  # type: ignore[misc,assignment]

# ---------------------------------------------------------------------------
# Model imports
# ---------------------------------------------------------------------------

from greenlang.agents.eudr.information_gathering.models import (
    CertificationBody,
    CertificateVerificationResult,
    CompletenessReport,
    EUDRCommodity,
    ExternalDatabaseSource,
    GatheringOperation,
    GatheringOperationStatus,
    HarvestResult,
    InformationPackage,
    QueryResult,
    SupplierProfile,
    Article9ElementStatus,
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

class InformationGatheringService:
    """High-level service facade for the Information Gathering Agent.

    Wires together all 7 processing engines and provides a unified API
    for information collection, verification, normalization, and packaging
    per EUDR Article 9 requirements.

    Attributes:
        config: Agent configuration.
        provenance: SHA-256 provenance tracker.
        _db_pool: PostgreSQL async connection pool.
        _redis: Redis async client.
        _engines: Dictionary of initialized engines.
        _initialized: Whether startup has completed.

    Example:
        >>> service = InformationGatheringService()
        >>> await service.startup()
        >>> operation = await service.gather_information(
        ...     operator_id="OP-001",
        ...     commodity=EUDRCommodity.COCOA,
        ... )
        >>> assert operation.status == GatheringOperationStatus.COMPLETED
    """

    def __init__(
        self,
        config: Optional[InformationGatheringConfig] = None,
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

        logger.info("InformationGatheringService created")

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
        logger.info("InformationGatheringService startup initiated")

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
                logger.warning("PostgreSQL pool init failed: %s", e)
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
                logger.warning("Redis init failed: %s", e)
                self._redis = None

        # Initialize engines
        self._init_engines()

        self._initialized = True
        elapsed = (time.monotonic() - start) * 1000
        engine_count = len(self._engines)

        logger.info(
            f"InformationGatheringService startup complete: "
            f"{engine_count}/7 engines in {elapsed:.1f}ms"
        )

    def _init_engines(self) -> None:
        """Initialize all 7 processing engines with graceful degradation."""
        engine_specs: List[Tuple[str, Any]] = [
            ("external_database_connector", ExternalDatabaseConnectorEngine),
            ("certification_verification", CertificationVerificationEngine),
            ("public_data_mining", PublicDataMiningEngine),
            ("supplier_information_aggregator", SupplierInformationAggregator),
            ("completeness_validation", CompletenessValidationEngine),
            ("data_normalization", DataNormalizationEngine),
            ("package_assembly", PackageAssemblyEngine),
        ]

        for name, engine_cls in engine_specs:
            if engine_cls is not None:
                try:
                    engine = engine_cls(config=self.config)
                    self._engines[name] = engine
                    logger.info("Engine '%s' initialized", name)
                except Exception as e:
                    logger.warning("Engine '%s' init failed: %s", name, e)
            else:
                logger.debug("Engine '%s' class not available", name)

    async def shutdown(self) -> None:
        """Gracefully shutdown all connections and engines.

        Closes database pool, Redis connection, and any
        engine-specific resources.
        """
        logger.info("InformationGatheringService shutdown initiated")

        # Shutdown engines with async shutdown methods
        for name, engine in self._engines.items():
            if hasattr(engine, "shutdown"):
                try:
                    await engine.shutdown()
                    logger.info("Engine '%s' shut down", name)
                except Exception as e:
                    logger.warning("Engine '%s' shutdown error: %s", name, e)

        # Close Redis
        if self._redis is not None:
            try:
                await self._redis.close()
                logger.info("Redis connection closed")
            except Exception as e:
                logger.warning("Redis close error: %s", e)

        # Close database pool
        if self._db_pool is not None:
            try:
                await self._db_pool.close()
                logger.info("PostgreSQL pool closed")
            except Exception as e:
                logger.warning("PostgreSQL pool close error: %s", e)

        self._initialized = False
        logger.info("InformationGatheringService shutdown complete")

    # ------------------------------------------------------------------
    # Full Pipeline
    # ------------------------------------------------------------------

    async def gather_information(
        self,
        operator_id: str,
        commodity: EUDRCommodity,
        sources: Optional[List[ExternalDatabaseSource]] = None,
    ) -> GatheringOperation:
        """Execute the full information gathering pipeline.

        Orchestrates all 7 engines in sequence:
            1. Query external databases for regulatory/trade data
            2. Verify certificates from certification bodies
            3. Harvest public datasets (FAO, GFW, WGI, CPI, etc.)
            4. Aggregate supplier profiles with entity resolution
            5. Validate Article 9 completeness
            6. Normalize data (units, dates, coordinates, currencies)
            7. Assemble information package for DDS submission

        Args:
            operator_id: EUDR operator identifier.
            commodity: Commodity being assessed.
            sources: Optional list of external sources to query.
                     If None, queries all enabled sources.

        Returns:
            GatheringOperation with status and package reference.
        """
        start = time.monotonic()
        operation_id = _new_uuid()

        logger.info(
            f"Starting information gathering {operation_id} for "
            f"operator={operator_id}, commodity={commodity.value}"
        )

        operation = GatheringOperation(
            operation_id=operation_id,
            operator_id=operator_id,
            commodity=commodity,
            status=GatheringOperationStatus.IN_PROGRESS,
        )

        try:
            # Step 1: Query external databases
            query_results = await self._step_query_external_databases(
                operation_id, commodity, sources
            )
            operation.total_records_collected = sum(
                qr.record_count for qr in query_results
            )
            operation.sources_completed = [
                qr.source.value for qr in query_results
            ]

            # Step 2: Verify certificates
            cert_results = await self._step_verify_certificates(
                operation_id, commodity, query_results
            )
            operation.total_certificates_verified = len(cert_results)

            # Step 3: Harvest public data
            harvest_results = await self._step_harvest_public_data(
                operation_id, commodity
            )

            # Step 4: Aggregate supplier profiles
            supplier_profiles = await self._step_aggregate_suppliers(
                operation_id, query_results
            )
            operation.total_suppliers_resolved = len(supplier_profiles)

            # Step 5: Validate completeness
            completeness_report = await self._step_validate_completeness(
                operation_id, commodity
            )
            operation.completeness_score = completeness_report.completeness_score
            operation.completeness_classification = (
                completeness_report.completeness_classification.value
            )

            # Step 6: Normalize data (handled within assembly)
            # Step 7: Assemble information package
            package = await self._step_assemble_package(
                operation_id,
                operator_id,
                commodity,
                query_results,
                cert_results,
                supplier_profiles,
                harvest_results,
                completeness_report,
            )
            operation.package_id = package.package_id

            # Mark operation complete
            operation.status = GatheringOperationStatus.COMPLETED
            operation.completed_at = utcnow()
            elapsed_ms = int((time.monotonic() - start) * 1000)
            operation.duration_ms = elapsed_ms
            operation.provenance_hash = _compute_hash(
                operation.model_dump(mode="json")
            )

            if METRICS_AVAILABLE and record_gathering_operation is not None:
                record_gathering_operation(commodity.value, "completed")

            logger.info(
                f"Information gathering {operation_id} completed in "
                f"{elapsed_ms}ms (completeness={operation.completeness_score}%)"
            )

        except Exception as e:
            operation.status = GatheringOperationStatus.FAILED
            operation.completed_at = utcnow()
            elapsed_ms = int((time.monotonic() - start) * 1000)
            operation.duration_ms = elapsed_ms

            if METRICS_AVAILABLE and record_api_error is not None:
                record_api_error("gather_information")

            logger.error(
                f"Information gathering {operation_id} failed: "
                f"{type(e).__name__}: {str(e)[:500]}",
                exc_info=True,
            )

        return operation

    # ------------------------------------------------------------------
    # Pipeline step helpers
    # ------------------------------------------------------------------

    async def _step_query_external_databases(
        self,
        operation_id: str,
        commodity: EUDRCommodity,
        sources: Optional[List[ExternalDatabaseSource]],
    ) -> List[QueryResult]:
        """Step 1: Query external regulatory/trade databases.

        Args:
            operation_id: Current operation identifier.
            commodity: Commodity for query filtering.
            sources: Specific sources or None for all enabled.

        Returns:
            List of query results from external databases.
        """
        engine = self._engines.get("external_database_connector")
        if engine is None:
            logger.warning("ExternalDatabaseConnectorEngine not available")
            return []

        target_sources = sources or list(ExternalDatabaseSource)
        results: List[QueryResult] = []

        for source in target_sources:
            try:
                result = await engine.query_source(
                    source, {"commodity": commodity.value}
                )
                results.append(result)
            except Exception as e:
                logger.warning(
                    f"External query {source.value} failed for "
                    f"operation {operation_id}: {e}"
                )

        return results

    async def _step_verify_certificates(
        self,
        operation_id: str,
        commodity: EUDRCommodity,
        query_results: List[QueryResult],
    ) -> List[CertificateVerificationResult]:
        """Step 2: Verify certificates discovered in external data.

        Args:
            operation_id: Current operation identifier.
            commodity: Commodity for body selection.
            query_results: Query results that may contain certificate IDs.

        Returns:
            List of certificate verification results.
        """
        engine = self._engines.get("certification_verification")
        if engine is None:
            logger.warning("CertificationVerificationEngine not available")
            return []

        # Extract certificate references from query results
        cert_refs: List[Tuple[str, CertificationBody]] = []
        for qr in query_results:
            for record in qr.records:
                cert_id = record.get("certificate_id") or record.get("cert_id")
                body_str = record.get("certification_body") or record.get("cert_body")
                if cert_id and body_str:
                    try:
                        body = CertificationBody(body_str)
                        cert_refs.append((cert_id, body))
                    except ValueError:
                        pass

        results: List[CertificateVerificationResult] = []
        for cert_id, body in cert_refs:
            try:
                result = await engine.verify_certificate(cert_id, body)
                results.append(result)
            except Exception as e:
                logger.warning(
                    f"Certificate verification {cert_id} failed for "
                    f"operation {operation_id}: {e}"
                )

        return results

    async def _step_harvest_public_data(
        self,
        operation_id: str,
        commodity: EUDRCommodity,
    ) -> List[HarvestResult]:
        """Step 3: Harvest publicly available data.

        Args:
            operation_id: Current operation identifier.
            commodity: Commodity for harvest filtering.

        Returns:
            List of harvest results from public data sources.
        """
        engine = self._engines.get("public_data_mining")
        if engine is None:
            logger.warning("PublicDataMiningEngine not available")
            return []

        results: List[HarvestResult] = []
        harvest_sources = [
            ExternalDatabaseSource.FAO_STAT,
            ExternalDatabaseSource.GLOBAL_FOREST_WATCH,
            ExternalDatabaseSource.WORLD_BANK_WGI,
            ExternalDatabaseSource.TRANSPARENCY_CPI,
        ]

        for source in harvest_sources:
            try:
                result = await engine.harvest_source(
                    source, commodity=commodity.value
                )
                results.append(result)
            except Exception as e:
                logger.warning(
                    f"Public data harvest {source.value} failed for "
                    f"operation {operation_id}: {e}"
                )

        return results

    async def _step_aggregate_suppliers(
        self,
        operation_id: str,
        query_results: List[QueryResult],
    ) -> List[SupplierProfile]:
        """Step 4: Aggregate supplier information from all sources.

        Args:
            operation_id: Current operation identifier.
            query_results: External database results with supplier data.

        Returns:
            List of aggregated supplier profiles.
        """
        engine = self._engines.get("supplier_information_aggregator")
        if engine is None:
            logger.warning("SupplierInformationAggregator not available")
            return []

        # Collect unique supplier IDs from query results
        supplier_ids: Dict[str, Dict[str, Any]] = {}
        for qr in query_results:
            for record in qr.records:
                sid = record.get("supplier_id")
                if sid and sid not in supplier_ids:
                    supplier_ids[sid] = {qr.source.value: record}

        profiles: List[SupplierProfile] = []
        for sid, sources in supplier_ids.items():
            try:
                profile = await engine.aggregate_supplier(sid, sources)
                profiles.append(profile)
            except Exception as e:
                logger.warning(
                    f"Supplier aggregation {sid} failed for "
                    f"operation {operation_id}: {e}"
                )

        return profiles

    async def _step_validate_completeness(
        self,
        operation_id: str,
        commodity: EUDRCommodity,
    ) -> CompletenessReport:
        """Step 5: Validate Article 9 information completeness.

        Args:
            operation_id: Current operation identifier.
            commodity: Commodity being assessed.

        Returns:
            CompletenessReport with element scores and gaps.
        """
        engine = self._engines.get("completeness_validation")
        if engine is None:
            logger.warning("CompletenessValidationEngine not available")
            return CompletenessReport(
                operation_id=operation_id,
                commodity=commodity,
            )

        try:
            report = await engine.validate_completeness(
                operation_id=operation_id,
                commodity=commodity,
            )
            return report
        except Exception as e:
            logger.warning(
                f"Completeness validation failed for operation "
                f"{operation_id}: {e}"
            )
            return CompletenessReport(
                operation_id=operation_id,
                commodity=commodity,
            )

    async def _step_assemble_package(
        self,
        operation_id: str,
        operator_id: str,
        commodity: EUDRCommodity,
        query_results: List[QueryResult],
        cert_results: List[CertificateVerificationResult],
        supplier_profiles: List[SupplierProfile],
        harvest_results: List[HarvestResult],
        completeness_report: CompletenessReport,
    ) -> InformationPackage:
        """Step 7: Assemble the information package.

        Args:
            operation_id: Current operation identifier.
            operator_id: EUDR operator identifier.
            commodity: Commodity being assessed.
            query_results: Results from external database queries.
            cert_results: Certificate verification results.
            supplier_profiles: Aggregated supplier profiles.
            harvest_results: Public data harvest results.
            completeness_report: Article 9 completeness report.

        Returns:
            Assembled InformationPackage for DDS submission.
        """
        engine = self._engines.get("package_assembly")
        if engine is not None:
            try:
                package = await engine.assemble_package(
                    operation_id=operation_id,
                    operator_id=operator_id,
                    commodity=commodity,
                    query_results=query_results,
                    cert_results=cert_results,
                    supplier_profiles=supplier_profiles,
                    harvest_results=harvest_results,
                    completeness_report=completeness_report,
                )
                return package
            except Exception as e:
                logger.warning(
                    f"Package assembly engine failed for operation "
                    f"{operation_id}: {e}"
                )

        # Fallback: build a minimal package inline
        logger.info(
            f"Assembling package inline for operation {operation_id}"
        )
        package_id = _new_uuid()

        # Group query results by source
        external_data: Dict[str, List[QueryResult]] = {}
        for qr in query_results:
            key = qr.source.value
            external_data.setdefault(key, []).append(qr)

        package = InformationPackage(
            package_id=package_id,
            operator_id=operator_id,
            commodity=commodity,
            completeness_score=completeness_report.completeness_score,
            completeness_classification=(
                completeness_report.completeness_classification.value
            ),
            supplier_profiles=supplier_profiles,
            external_data=external_data,
            certification_results=cert_results,
            gap_report=completeness_report.gap_report,
        )
        package.package_hash = _compute_hash(
            package.model_dump(mode="json")
        )

        if METRICS_AVAILABLE and observe_package_assembly_duration is not None:
            observe_package_assembly_duration(0.0)

        return package

    # ------------------------------------------------------------------
    # Individual engine delegates
    # ------------------------------------------------------------------

    async def query_external_database(
        self,
        source: ExternalDatabaseSource,
        params: Dict[str, Any],
    ) -> QueryResult:
        """Query a specific external database.

        Delegates to ExternalDatabaseConnectorEngine.

        Args:
            source: Database source to query.
            params: Source-specific query parameters.

        Returns:
            QueryResult with records and provenance hash.

        Raises:
            RuntimeError: If engine is not available.
        """
        engine = self._engines.get("external_database_connector")
        if engine is None:
            raise RuntimeError("ExternalDatabaseConnectorEngine not available")
        return await engine.query_source(source, params)

    async def verify_certificate(
        self,
        cert_id: str,
        body: CertificationBody,
    ) -> CertificateVerificationResult:
        """Verify a single certificate against its official registry.

        Delegates to CertificationVerificationEngine.

        Args:
            cert_id: Certificate identifier.
            body: Certification body that issued the certificate.

        Returns:
            CertificateVerificationResult with status and provenance.

        Raises:
            RuntimeError: If engine is not available.
        """
        engine = self._engines.get("certification_verification")
        if engine is None:
            raise RuntimeError("CertificationVerificationEngine not available")
        return await engine.verify_certificate(cert_id, body)

    async def batch_verify_certificates(
        self,
        certs: List[Tuple[str, CertificationBody]],
    ) -> List[CertificateVerificationResult]:
        """Verify multiple certificates in batch.

        Delegates to CertificationVerificationEngine batch method.

        Args:
            certs: List of (certificate_id, certification_body) tuples.

        Returns:
            List of CertificateVerificationResult objects.

        Raises:
            RuntimeError: If engine is not available.
        """
        engine = self._engines.get("certification_verification")
        if engine is None:
            raise RuntimeError("CertificationVerificationEngine not available")
        return await engine.batch_verify(certs)

    async def harvest_public_data(
        self,
        source: ExternalDatabaseSource,
        country_code: Optional[str] = None,
        commodity: Optional[str] = None,
    ) -> HarvestResult:
        """Harvest public data from a specific source.

        Delegates to PublicDataMiningEngine.

        Args:
            source: Data source to harvest.
            country_code: Optional ISO 3166-1 alpha-2 country code.
            commodity: Optional commodity name.

        Returns:
            HarvestResult with harvested data and freshness status.

        Raises:
            RuntimeError: If engine is not available.
        """
        engine = self._engines.get("public_data_mining")
        if engine is None:
            raise RuntimeError("PublicDataMiningEngine not available")
        return await engine.harvest_source(
            source, country_code=country_code, commodity=commodity
        )

    async def aggregate_supplier(
        self,
        supplier_id: str,
        sources: Dict[str, Any],
    ) -> SupplierProfile:
        """Aggregate supplier information from multiple sources.

        Delegates to SupplierInformationAggregator.

        Args:
            supplier_id: Supplier identifier.
            sources: Dictionary of source_name -> source_data.

        Returns:
            Unified SupplierProfile with completeness and confidence scores.

        Raises:
            RuntimeError: If engine is not available.
        """
        engine = self._engines.get("supplier_information_aggregator")
        if engine is None:
            raise RuntimeError("SupplierInformationAggregator not available")
        return await engine.aggregate_supplier(supplier_id, sources)

    async def validate_completeness(
        self,
        operation_id: str,
        commodity: EUDRCommodity,
        elements: Optional[List[Article9ElementStatus]] = None,
        is_simplified_dd: bool = False,
    ) -> CompletenessReport:
        """Validate Article 9 information completeness.

        Delegates to CompletenessValidationEngine.

        Args:
            operation_id: Current operation identifier.
            commodity: Commodity being assessed.
            elements: Optional pre-populated element statuses.
            is_simplified_dd: Whether simplified due diligence applies.

        Returns:
            CompletenessReport with element scores and gap analysis.

        Raises:
            RuntimeError: If engine is not available.
        """
        engine = self._engines.get("completeness_validation")
        if engine is None:
            raise RuntimeError("CompletenessValidationEngine not available")
        return await engine.validate_completeness(
            operation_id=operation_id,
            commodity=commodity,
            elements=elements,
            is_simplified_dd=is_simplified_dd,
        )

    async def assemble_package(
        self,
        operation_id: str,
        operator_id: str,
        commodity: EUDRCommodity,
        query_results: Optional[List[QueryResult]] = None,
        cert_results: Optional[List[CertificateVerificationResult]] = None,
        supplier_profiles: Optional[List[SupplierProfile]] = None,
    ) -> InformationPackage:
        """Assemble an information package for DDS submission.

        Delegates to PackageAssemblyEngine.

        Args:
            operation_id: Current operation identifier.
            operator_id: EUDR operator identifier.
            commodity: Commodity being assessed.
            query_results: Optional external database results.
            cert_results: Optional certificate verification results.
            supplier_profiles: Optional aggregated supplier profiles.

        Returns:
            Assembled InformationPackage with integrity hash.

        Raises:
            RuntimeError: If engine is not available.
        """
        engine = self._engines.get("package_assembly")
        if engine is None:
            raise RuntimeError("PackageAssemblyEngine not available")
        return await engine.assemble_package(
            operation_id=operation_id,
            operator_id=operator_id,
            commodity=commodity,
            query_results=query_results or [],
            cert_results=cert_results or [],
            supplier_profiles=supplier_profiles or [],
        )

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
            "agent_id": "GL-EUDR-IGA-027",
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
            "external_database_connector",
            "certification_verification",
            "public_data_mining",
            "supplier_information_aggregator",
            "completeness_validation",
            "data_normalization",
            "package_assembly",
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
            name: Engine name (e.g., 'external_database_connector').

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
_service_instance: Optional[InformationGatheringService] = None

def get_service(
    config: Optional[InformationGatheringConfig] = None,
) -> InformationGatheringService:
    """Get the global InformationGatheringService singleton instance.

    Thread-safe lazy initialization. Creates a new service instance
    on first call.

    Args:
        config: Optional configuration override for first creation.

    Returns:
        InformationGatheringService singleton instance.

    Example:
        >>> service = get_service()
        >>> assert service is get_service()
    """
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = InformationGatheringService(config)
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
        >>> from greenlang.agents.eudr.information_gathering.setup import lifespan
        >>> app = FastAPI(lifespan=lifespan)

    Args:
        app: FastAPI application instance.

    Yields:
        None - application runs between startup and shutdown.
    """
    service = get_service()
    await service.startup()
    logger.info("Information Gathering Agent lifespan: startup complete")

    try:
        yield
    finally:
        await service.shutdown()
        logger.info("Information Gathering Agent lifespan: shutdown complete")
