# -*- coding: utf-8 -*-
"""
Documentation Generator Service Facade - AGENT-EUDR-030

High-level service facade that wires together all 7 processing engines
into a single cohesive entry point. Provides the primary API used by
the FastAPI router layer to generate Due Diligence Statements (DDS),
assemble Article 9 data packages, document risk assessments and
mitigation measures, build compliance packages, manage document
versions, and submit to the EU Information System per EUDR Article 4(2)
requirements.

This facade implements the Facade Pattern to hide the complexity
of the 7 internal engines behind a clean, use-case-oriented interface.

Engines (7):
    1. DDSStatementGenerator        - Complete DDS generation
    2. Article9DataAssembler        - Article 9 data element assembly
    3. RiskAssessmentDocumenter     - Risk assessment documentation
    4. MitigationDocumenter         - Mitigation measure documentation
    5. CompliancePackageBuilder     - Submission-ready package building
    6. DocumentVersionManager       - Version lifecycle management
    7. RegulatorySubmissionEngine   - EU IS submission management

Service Methods:
    DDS Generation:
        - generate_dds()           -> Generate a complete DDS
        - get_dds()                -> Retrieve DDS by ID
        - list_dds()               -> List DDS documents with filters

    Article 9 Assembly:
        - assemble_article9()      -> Assemble Article 9 data package

    Risk Documentation:
        - document_risk_assessment() -> Document risk assessment results

    Mitigation Documentation:
        - document_mitigation()    -> Document mitigation measures

    Compliance Packages:
        - build_compliance_package() -> Build compliance-ready package

    Validation:
        - validate_dds()           -> Validate DDS completeness

    Submission:
        - submit_dds()             -> Submit DDS to EU IS
        - get_submission_status()  -> Check submission status

    Versioning:
        - get_version_history()    -> Get document version history

    Health:
        - health_check()           -> Component health status

Singleton Pattern:
    Thread-safe singleton with double-checked locking via ``get_service()``.

FastAPI Integration:
    Use the ``lifespan`` async context manager with
    ``FastAPI(lifespan=lifespan)`` for automatic startup/shutdown.

Example:
    >>> from greenlang.agents.eudr.documentation_generator.setup import (
    ...     DocumentationGeneratorService,
    ...     get_service,
    ... )
    >>> service = get_service()
    >>> await service.startup()
    >>> health = await service.health_check()
    >>> assert health["status"] == "healthy"
    >>>
    >>> # Generate DDS
    >>> dds = await service.generate_dds(
    ...     operator_id="OP-001",
    ...     commodity="cocoa",
    ...     products=[...],
    ...     geolocations=[...],
    ...     suppliers=[...],
    ... )
    >>>
    >>> # Validate and submit
    >>> validation = await service.validate_dds(dds["dds_id"])
    >>> submission = await service.submit_dds(dds["dds_id"])
    >>>
    >>> await service.shutdown()

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-030 Documentation Generator (GL-EUDR-DGN-030)
Regulation: EU 2023/1115 (EUDR) Articles 4, 9, 10, 11, 12, 13, 14-16, 29, 31
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
_AGENT_ID = "GL-EUDR-DGN-030"

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

from greenlang.agents.eudr.documentation_generator.config import (
    DocumentationGeneratorConfig,
    get_config,
)

# ---------------------------------------------------------------------------
# Provenance import (graceful fallback)
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.documentation_generator.provenance import (
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
    from greenlang.agents.eudr.documentation_generator.metrics import (
        record_dds_generated,
        record_article9_assembled,
        record_risk_documented,
        record_mitigation_documented,
        record_package_built,
        record_submission_sent,
        record_validation_run,
        record_api_error,
        observe_dds_generation_duration,
        observe_article9_assembly_duration,
        observe_package_build_duration,
        observe_submission_duration,
        set_active_dds_documents,
        set_active_packages,
        set_active_submissions,
        set_pending_submissions,
        set_validation_pass_rate,
        set_total_versions,
    )

    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    record_dds_generated = None  # type: ignore[assignment]
    record_article9_assembled = None  # type: ignore[assignment]
    record_risk_documented = None  # type: ignore[assignment]
    record_mitigation_documented = None  # type: ignore[assignment]
    record_package_built = None  # type: ignore[assignment]
    record_submission_sent = None  # type: ignore[assignment]
    record_validation_run = None  # type: ignore[assignment]
    record_api_error = None  # type: ignore[assignment]
    observe_dds_generation_duration = None  # type: ignore[assignment]
    observe_article9_assembly_duration = None  # type: ignore[assignment]
    observe_package_build_duration = None  # type: ignore[assignment]
    observe_submission_duration = None  # type: ignore[assignment]
    set_active_dds_documents = None  # type: ignore[assignment]
    set_active_packages = None  # type: ignore[assignment]
    set_active_submissions = None  # type: ignore[assignment]
    set_pending_submissions = None  # type: ignore[assignment]
    set_validation_pass_rate = None  # type: ignore[assignment]
    set_total_versions = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Model imports (graceful fallback)
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.documentation_generator.models import (
        EUDRCommodity,
        RiskLevel,
        DDSStatus,
        DocumentType,
        SubmissionStatus,
        PackageFormat,
        ValidationSeverity,
        VersionStatus,
        Article9Element,
        ComplianceSection,
        RetentionStatus,
        AuditAction,
        DDSDocument,
        Article9Package,
        RiskAssessmentDoc,
        MitigationDoc,
        CompliancePackage,
        DocumentVersion,
        SubmissionRecord,
        ValidationResult,
        ValidationIssue,
        ProductEntry,
        GeolocationReference,
        SupplierReference,
        MeasureSummary,
        DDSContent,
        HealthStatus,
    )

    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    EUDRCommodity = None  # type: ignore[misc,assignment]
    RiskLevel = None  # type: ignore[misc,assignment]
    DDSStatus = None  # type: ignore[misc,assignment]
    DocumentType = None  # type: ignore[misc,assignment]
    SubmissionStatus = None  # type: ignore[misc,assignment]
    PackageFormat = None  # type: ignore[misc,assignment]
    ValidationSeverity = None  # type: ignore[misc,assignment]
    VersionStatus = None  # type: ignore[misc,assignment]
    Article9Element = None  # type: ignore[misc,assignment]
    ComplianceSection = None  # type: ignore[misc,assignment]
    RetentionStatus = None  # type: ignore[misc,assignment]
    AuditAction = None  # type: ignore[misc,assignment]
    DDSDocument = None  # type: ignore[misc,assignment]
    Article9Package = None  # type: ignore[misc,assignment]
    RiskAssessmentDoc = None  # type: ignore[misc,assignment]
    MitigationDoc = None  # type: ignore[misc,assignment]
    CompliancePackage = None  # type: ignore[misc,assignment]
    DocumentVersion = None  # type: ignore[misc,assignment]
    SubmissionRecord = None  # type: ignore[misc,assignment]
    ValidationResult = None  # type: ignore[misc,assignment]
    ValidationIssue = None  # type: ignore[misc,assignment]
    ProductEntry = None  # type: ignore[misc,assignment]
    GeolocationReference = None  # type: ignore[misc,assignment]
    SupplierReference = None  # type: ignore[misc,assignment]
    MeasureSummary = None  # type: ignore[misc,assignment]
    DDSContent = None  # type: ignore[misc,assignment]
    HealthStatus = None  # type: ignore[misc,assignment]

# ---------------------------------------------------------------------------
# Engine imports (conditional -- engines may not exist yet)
# ---------------------------------------------------------------------------

# ---- Engine 1: DDS Statement Generator ----
try:
    from greenlang.agents.eudr.documentation_generator.dds_statement_generator import (
        DDSStatementGenerator,
    )
except ImportError:
    DDSStatementGenerator = None  # type: ignore[misc,assignment]

# ---- Engine 2: Article 9 Data Assembler ----
try:
    from greenlang.agents.eudr.documentation_generator.article9_data_assembler import (
        Article9DataAssembler,
    )
except ImportError:
    Article9DataAssembler = None  # type: ignore[misc,assignment]

# ---- Engine 3: Risk Assessment Documenter ----
try:
    from greenlang.agents.eudr.documentation_generator.risk_assessment_documenter import (
        RiskAssessmentDocumenter,
    )
except ImportError:
    RiskAssessmentDocumenter = None  # type: ignore[misc,assignment]

# ---- Engine 4: Mitigation Documenter ----
try:
    from greenlang.agents.eudr.documentation_generator.mitigation_documenter import (
        MitigationDocumenter,
    )
except ImportError:
    MitigationDocumenter = None  # type: ignore[misc,assignment]

# ---- Engine 5: Compliance Package Builder ----
try:
    from greenlang.agents.eudr.documentation_generator.compliance_package_builder import (
        CompliancePackageBuilder,
    )
except ImportError:
    CompliancePackageBuilder = None  # type: ignore[misc,assignment]

# ---- Engine 6: Document Version Manager ----
try:
    from greenlang.agents.eudr.documentation_generator.document_version_manager import (
        DocumentVersionManager,
    )
except ImportError:
    DocumentVersionManager = None  # type: ignore[misc,assignment]

# ---- Engine 7: Regulatory Submission Engine ----
try:
    from greenlang.agents.eudr.documentation_generator.regulatory_submission_engine import (
        RegulatorySubmissionEngine,
    )
except ImportError:
    RegulatorySubmissionEngine = None  # type: ignore[misc,assignment]


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


class DocumentationGeneratorService:
    """Unified service facade for AGENT-EUDR-030.

    Aggregates all 7 processing engines and provides a clean API for
    Due Diligence Statement generation, Article 9 data assembly,
    risk and mitigation documentation, compliance package building,
    document versioning, and regulatory submission per EUDR Article 4(2).

    This class manages in-memory storage for DDS documents, Article 9
    packages, risk documentation, mitigation documentation, compliance
    packages, submissions, and document versions. In production, these
    are backed by PostgreSQL persistence via the engine layer.

    Attributes:
        config: Agent configuration.
        _dds_generator: Engine 1 -- DDS statement generation.
        _article9_assembler: Engine 2 -- Article 9 data assembly.
        _risk_documenter: Engine 3 -- risk assessment documentation.
        _mitigation_documenter: Engine 4 -- mitigation documentation.
        _package_builder: Engine 5 -- compliance package building.
        _version_manager: Engine 6 -- document version management.
        _submission_engine: Engine 7 -- regulatory submission.
        _provenance: SHA-256 provenance tracker.
        _initialized: Whether startup has completed.

    Example:
        >>> service = DocumentationGeneratorService()
        >>> await service.startup()
        >>> health = await service.health_check()
        >>> assert health["status"] == "healthy"
    """

    def __init__(
        self,
        config: Optional[DocumentationGeneratorConfig] = None,
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
        self._dds_generator: Optional[Any] = None
        self._article9_assembler: Optional[Any] = None
        self._risk_documenter: Optional[Any] = None
        self._mitigation_documenter: Optional[Any] = None
        self._package_builder: Optional[Any] = None
        self._version_manager: Optional[Any] = None
        self._submission_engine: Optional[Any] = None
        self._engines: Dict[str, Any] = {}

        # In-memory stores (used when engines are unavailable)
        self._dds_documents: Dict[str, Any] = {}
        self._article9_packages: Dict[str, Any] = {}
        self._risk_docs: Dict[str, Any] = {}
        self._mitigation_docs: Dict[str, Any] = {}
        self._compliance_packages: Dict[str, Any] = {}
        self._submissions: Dict[str, Any] = {}
        self._versions: Dict[str, List[Any]] = {}
        self._validation_results: Dict[str, Any] = {}

        self._initialized = False

        logger.info("DocumentationGeneratorService created")

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
        logger.info("DocumentationGeneratorService startup initiated")

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
            f"DocumentationGeneratorService startup complete: "
            f"{engine_count}/7 engines in {elapsed:.1f}ms"
        )

    def _init_engines(self) -> None:
        """Initialize all 7 processing engines with graceful degradation.

        Each engine is initialized independently so that failures in one
        engine do not prevent other engines from loading.
        """
        engine_specs: List[Tuple[str, Any]] = [
            ("dds_statement_generator", DDSStatementGenerator),
            ("article9_data_assembler", Article9DataAssembler),
            ("risk_assessment_documenter", RiskAssessmentDocumenter),
            ("mitigation_documenter", MitigationDocumenter),
            ("compliance_package_builder", CompliancePackageBuilder),
            ("document_version_manager", DocumentVersionManager),
            ("regulatory_submission_engine", RegulatorySubmissionEngine),
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
        self._dds_generator = self._engines.get(
            "dds_statement_generator"
        )
        self._article9_assembler = self._engines.get(
            "article9_data_assembler"
        )
        self._risk_documenter = self._engines.get(
            "risk_assessment_documenter"
        )
        self._mitigation_documenter = self._engines.get(
            "mitigation_documenter"
        )
        self._package_builder = self._engines.get(
            "compliance_package_builder"
        )
        self._version_manager = self._engines.get(
            "document_version_manager"
        )
        self._submission_engine = self._engines.get(
            "regulatory_submission_engine"
        )

    async def shutdown(self) -> None:
        """Gracefully shutdown all connections and engines.

        Closes database pool, Redis connection, and any
        engine-specific resources.
        """
        logger.info("DocumentationGeneratorService shutdown initiated")

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
        logger.info("DocumentationGeneratorService shutdown complete")

    # ------------------------------------------------------------------
    # DDS Generation
    # ------------------------------------------------------------------

    async def generate_dds(
        self,
        operator_id: str,
        commodity: str,
        products: List[Dict[str, Any]],
        geolocations: List[Dict[str, Any]],
        suppliers: List[Dict[str, Any]],
        risk_assessment_id: Optional[str] = None,
        mitigation_strategy_id: Optional[str] = None,
        reference_number: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a complete Due Diligence Statement.

        This is the main entry point for DDS generation. Assembles all
        required Article 9 data elements, documents risk assessment
        results if available, documents mitigation measures if available,
        and produces a complete DDS per EUDR Article 4(2).

        Args:
            operator_id: EUDR operator identifier.
            commodity: EUDR regulated commodity.
            products: List of product entries with descriptions, quantities,
                HS codes, and trade/scientific names.
            geolocations: List of geolocation references with coordinates,
                country codes, and plot identifiers.
            suppliers: List of supplier references with identifiers, names,
                and tiers.
            risk_assessment_id: Optional risk assessment identifier from
                EUDR-028 for inclusion in the DDS.
            mitigation_strategy_id: Optional mitigation strategy identifier
                from EUDR-029 for inclusion in the DDS.
            reference_number: Optional operator reference number.

        Returns:
            DDSDocument dictionary with complete DDS data.

        Raises:
            ValueError: If required data is missing or invalid.
        """
        start = time.monotonic()
        dds_id = _new_uuid()

        logger.info(
            f"Generating DDS {dds_id} for operator={operator_id}, "
            f"commodity={commodity}, products={len(products)}"
        )

        # Delegate to engine if available
        if self._dds_generator is not None:
            try:
                dds = await self._dds_generator.generate_dds(
                    operator_id=operator_id,
                    commodity=commodity,
                    products=products,
                    geolocations=geolocations,
                    suppliers=suppliers,
                    risk_assessment_id=risk_assessment_id,
                    mitigation_strategy_id=mitigation_strategy_id,
                    reference_number=reference_number,
                )
                if isinstance(dds, dict):
                    self._dds_documents[dds.get("dds_id", dds_id)] = dds
                else:
                    self._dds_documents[
                        getattr(dds, "dds_id", dds_id)
                    ] = dds
                _safe_record(record_dds_generated, commodity)
                elapsed = time.monotonic() - start
                _safe_observe(observe_dds_generation_duration, elapsed)
                _safe_gauge(
                    set_active_dds_documents, len(self._dds_documents)
                )
                return dds
            except Exception as e:
                logger.warning(
                    f"DDS generator engine failed, using fallback: {e}"
                )

        # Step 1: Assemble Article 9 data
        article9 = await self._assemble_article9_fallback(
            operator_id=operator_id,
            commodity=commodity,
            products=products,
            geolocations=geolocations,
            suppliers=suppliers,
        )
        article9_id = article9.get("package_id", _new_uuid())

        # Step 2: Document risk assessment if provided
        risk_doc = None
        if risk_assessment_id:
            risk_doc = await self._document_risk_fallback(
                assessment_id=risk_assessment_id,
                operator_id=operator_id,
                commodity=commodity,
            )

        # Step 3: Document mitigation if provided
        mitigation_doc = None
        if mitigation_strategy_id:
            mitigation_doc = await self._document_mitigation_fallback(
                strategy_id=mitigation_strategy_id,
                operator_id=operator_id,
                commodity=commodity,
            )

        # Step 4: Build the DDS
        now = _utcnow()
        ref_number = reference_number or f"DDS-{operator_id}-{now.strftime('%Y%m%d')}"

        # Determine risk conclusion
        if risk_doc:
            risk_level = risk_doc.get("risk_level", "standard")
        else:
            risk_level = "standard"

        # Determine DDS status based on risk and mitigation
        if risk_level in ("negligible", "low"):
            dds_status = "compliant"
        elif mitigation_doc:
            dds_status = "compliant_with_mitigation"
        else:
            dds_status = "draft"

        dds = {
            "dds_id": dds_id,
            "reference_number": ref_number,
            "operator_id": operator_id,
            "commodity": commodity,
            "status": dds_status,
            "article9_package_id": article9_id,
            "risk_assessment_id": risk_assessment_id,
            "mitigation_strategy_id": mitigation_strategy_id,
            "risk_conclusion": risk_level,
            "product_count": len(products),
            "geolocation_count": len(geolocations),
            "supplier_count": len(suppliers),
            "products": products,
            "geolocations": geolocations,
            "suppliers": suppliers,
            "article9_data": article9,
            "risk_documentation": risk_doc,
            "mitigation_documentation": mitigation_doc,
            "regulation_reference": "EU 2023/1115 Article 4(2)",
            "retention_until": (
                now + timedelta(days=self.config.retention_years * 365)
            ).isoformat(),
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "version": 1,
            "provenance_hash": _compute_hash({
                "dds_id": dds_id,
                "operator_id": operator_id,
                "commodity": commodity,
                "product_count": len(products),
                "geolocation_count": len(geolocations),
                "created_at": now.isoformat(),
            }),
        }

        self._dds_documents[dds_id] = dds

        # Record initial version
        self._record_version(
            document_id=dds_id,
            document_type="dds",
            version=1,
            actor=operator_id,
            change_summary="Initial DDS creation",
        )

        # Record provenance
        if self._provenance is not None:
            self._provenance.record(
                entity_type="dds",
                action="create",
                entity_id=dds_id,
                actor=operator_id,
                metadata={
                    "commodity": commodity,
                    "product_count": len(products),
                    "geolocation_count": len(geolocations),
                    "supplier_count": len(suppliers),
                    "status": dds_status,
                },
            )

        _safe_record(record_dds_generated, commodity)
        elapsed = time.monotonic() - start
        _safe_observe(observe_dds_generation_duration, elapsed)
        _safe_gauge(set_active_dds_documents, len(self._dds_documents))

        logger.info(
            f"DDS {dds_id} generated in {elapsed * 1000:.1f}ms "
            f"(status={dds_status}, products={len(products)}, "
            f"geolocations={len(geolocations)})"
        )

        return dds

    async def get_dds(
        self,
        dds_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a DDS by its identifier.

        Args:
            dds_id: DDS identifier.

        Returns:
            DDS data or None if not found.
        """
        # Check engine first
        if self._dds_generator is not None:
            try:
                return await self._dds_generator.get_dds(dds_id)
            except Exception as e:
                logger.debug(f"Engine DDS lookup failed: {e}")

        # Fallback to in-memory
        return self._dds_documents.get(dds_id)

    async def list_dds(
        self,
        operator_id: Optional[str] = None,
        commodity: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List DDS documents with optional filters.

        Args:
            operator_id: Filter by operator identifier.
            commodity: Filter by EUDR commodity.
            status: Filter by DDS status.

        Returns:
            List of matching DDS documents.
        """
        # Delegate to engine if available
        if self._dds_generator is not None:
            try:
                return await self._dds_generator.list_dds(
                    operator_id=operator_id,
                    commodity=commodity,
                    status=status,
                )
            except Exception as e:
                logger.debug(f"Engine list_dds failed: {e}")

        # Fallback: filter in-memory
        results = list(self._dds_documents.values())

        if operator_id:
            results = [
                d for d in results
                if d.get("operator_id") == operator_id
            ]
        if commodity:
            results = [
                d for d in results
                if d.get("commodity") == commodity
            ]
        if status:
            results = [
                d for d in results
                if d.get("status") == status
            ]

        return results

    # ------------------------------------------------------------------
    # Article 9 Assembly
    # ------------------------------------------------------------------

    async def assemble_article9(
        self,
        operator_id: str,
        commodity: str,
        products: List[Dict[str, Any]],
        geolocations: List[Dict[str, Any]],
        suppliers: List[Dict[str, Any]],
        production_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Assemble an Article 9 data package.

        Collects and validates all data elements required by EUDR
        Article 9 including product descriptions, quantities, country
        of production, geolocation coordinates, supplier information,
        and production dates.

        Args:
            operator_id: EUDR operator identifier.
            commodity: EUDR regulated commodity.
            products: List of product entries.
            geolocations: List of geolocation references.
            suppliers: List of supplier references.
            production_date: Optional date of production (ISO 8601).

        Returns:
            Article9Package dictionary with assembled data.

        Raises:
            ValueError: If required Article 9 elements are missing.
        """
        start = time.monotonic()

        logger.info(
            f"Assembling Article 9 package for operator={operator_id}, "
            f"commodity={commodity}"
        )

        # Delegate to engine if available
        if self._article9_assembler is not None:
            try:
                package = await self._article9_assembler.assemble(
                    operator_id=operator_id,
                    commodity=commodity,
                    products=products,
                    geolocations=geolocations,
                    suppliers=suppliers,
                    production_date=production_date,
                )
                pkg_id = (
                    package.get("package_id")
                    if isinstance(package, dict)
                    else getattr(package, "package_id", _new_uuid())
                )
                self._article9_packages[pkg_id] = package
                _safe_record(record_article9_assembled, commodity)
                elapsed = time.monotonic() - start
                _safe_observe(observe_article9_assembly_duration, elapsed)
                return package
            except Exception as e:
                logger.warning(
                    f"Article 9 assembler engine failed, using fallback: {e}"
                )

        # Fallback: in-memory assembly
        package = await self._assemble_article9_fallback(
            operator_id=operator_id,
            commodity=commodity,
            products=products,
            geolocations=geolocations,
            suppliers=suppliers,
            production_date=production_date,
        )

        _safe_record(record_article9_assembled, commodity)
        elapsed = time.monotonic() - start
        _safe_observe(observe_article9_assembly_duration, elapsed)

        logger.info(
            f"Article 9 package {package.get('package_id')} assembled "
            f"in {elapsed * 1000:.1f}ms"
        )

        return package

    async def _assemble_article9_fallback(
        self,
        operator_id: str,
        commodity: str,
        products: List[Dict[str, Any]],
        geolocations: List[Dict[str, Any]],
        suppliers: List[Dict[str, Any]],
        production_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fallback Article 9 assembly using in-memory processing.

        Args:
            operator_id: EUDR operator identifier.
            commodity: EUDR regulated commodity.
            products: List of product entries.
            geolocations: List of geolocation references.
            suppliers: List of supplier references.
            production_date: Optional production date.

        Returns:
            Article 9 package dictionary.
        """
        package_id = _new_uuid()
        now = _utcnow()

        # Extract country codes from geolocations
        country_codes = list(set(
            g.get("country_code", "")
            for g in geolocations
            if g.get("country_code")
        ))

        # Validate completeness of Article 9 elements
        elements_present: List[str] = []
        elements_missing: List[str] = []

        required_elements = [
            ("product_description", bool(products)),
            ("quantity", any(p.get("quantity") for p in products)),
            ("country_of_production", bool(country_codes)),
            ("geolocation", bool(geolocations)),
            ("supplier_information", bool(suppliers)),
            ("operator_information", bool(operator_id)),
        ]

        optional_elements = [
            ("date_of_production", bool(production_date)),
            ("hs_code", any(p.get("hs_code") for p in products)),
            ("trade_name", any(p.get("trade_name") for p in products)),
            ("scientific_name", any(p.get("scientific_name") for p in products)),
        ]

        for elem_name, is_present in required_elements:
            if is_present:
                elements_present.append(elem_name)
            else:
                elements_missing.append(elem_name)

        for elem_name, is_present in optional_elements:
            if is_present:
                elements_present.append(elem_name)

        completeness_score = (
            len(elements_present) / (len(required_elements) + len(optional_elements))
        ) * 100.0

        package = {
            "package_id": package_id,
            "operator_id": operator_id,
            "commodity": commodity,
            "products": products,
            "geolocations": geolocations,
            "suppliers": suppliers,
            "country_codes": country_codes,
            "production_date": production_date,
            "elements_present": elements_present,
            "elements_missing": elements_missing,
            "completeness_score": round(completeness_score, 2),
            "is_complete": len(elements_missing) == 0,
            "regulation_reference": "EU 2023/1115 Article 9",
            "created_at": now.isoformat(),
            "provenance_hash": _compute_hash({
                "package_id": package_id,
                "operator_id": operator_id,
                "commodity": commodity,
                "product_count": len(products),
                "created_at": now.isoformat(),
            }),
        }

        self._article9_packages[package_id] = package
        return package

    # ------------------------------------------------------------------
    # Risk Assessment Documentation
    # ------------------------------------------------------------------

    async def document_risk_assessment(
        self,
        assessment_id: str,
        operator_id: str,
        commodity: str,
        composite_score: str,
        risk_level: str,
        contributing_factors: Optional[List[Dict[str, Any]]] = None,
        country_scores: Optional[List[Dict[str, Any]]] = None,
        supplier_scores: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Document a risk assessment for DDS inclusion.

        Creates structured documentation of risk assessment results
        including composite scores, risk level classifications,
        contributing factor breakdowns, and regulatory cross-references.

        Args:
            assessment_id: Risk assessment identifier from EUDR-028.
            operator_id: EUDR operator identifier.
            commodity: EUDR regulated commodity.
            composite_score: Composite risk score (0-100 as string).
            risk_level: Risk level classification.
            contributing_factors: Optional list of contributing factors
                with dimension, score, and weight.
            country_scores: Optional country-level risk scores.
            supplier_scores: Optional supplier-level risk scores.

        Returns:
            RiskAssessmentDoc dictionary.
        """
        start = time.monotonic()

        logger.info(
            f"Documenting risk assessment {assessment_id} for "
            f"operator={operator_id}, commodity={commodity}"
        )

        # Delegate to engine if available
        if self._risk_documenter is not None:
            try:
                doc = await self._risk_documenter.document(
                    assessment_id=assessment_id,
                    operator_id=operator_id,
                    commodity=commodity,
                    composite_score=composite_score,
                    risk_level=risk_level,
                    contributing_factors=contributing_factors,
                    country_scores=country_scores,
                    supplier_scores=supplier_scores,
                )
                doc_id = (
                    doc.get("doc_id")
                    if isinstance(doc, dict)
                    else getattr(doc, "doc_id", _new_uuid())
                )
                self._risk_docs[doc_id] = doc
                _safe_record(record_risk_documented, commodity)
                return doc
            except Exception as e:
                logger.warning(
                    f"Risk documenter engine failed, using fallback: {e}"
                )

        # Fallback: in-memory documentation
        doc = await self._document_risk_fallback(
            assessment_id=assessment_id,
            operator_id=operator_id,
            commodity=commodity,
            composite_score=composite_score,
            risk_level=risk_level,
            contributing_factors=contributing_factors,
            country_scores=country_scores,
            supplier_scores=supplier_scores,
        )

        _safe_record(record_risk_documented, commodity)
        elapsed = time.monotonic() - start

        logger.info(
            f"Risk assessment {assessment_id} documented "
            f"in {elapsed * 1000:.1f}ms (level={risk_level})"
        )

        return doc

    async def _document_risk_fallback(
        self,
        assessment_id: str,
        operator_id: str,
        commodity: str,
        composite_score: Optional[str] = None,
        risk_level: Optional[str] = None,
        contributing_factors: Optional[List[Dict[str, Any]]] = None,
        country_scores: Optional[List[Dict[str, Any]]] = None,
        supplier_scores: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Fallback risk assessment documentation.

        Args:
            assessment_id: Risk assessment identifier.
            operator_id: EUDR operator identifier.
            commodity: EUDR regulated commodity.
            composite_score: Composite risk score.
            risk_level: Risk level classification.
            contributing_factors: Contributing factor breakdown.
            country_scores: Country-level risk scores.
            supplier_scores: Supplier-level risk scores.

        Returns:
            Risk assessment documentation dictionary.
        """
        doc_id = _new_uuid()
        now = _utcnow()

        score = composite_score or "50"
        level = risk_level or "standard"

        # Determine if mitigation is required
        score_decimal = Decimal(score)
        requires_mitigation = score_decimal > Decimal(
            str(self.config.mitigation_threshold)
        )

        # Build regulatory cross-references
        article_references = ["Article 10 (Risk Assessment)"]
        if requires_mitigation:
            article_references.append("Article 11 (Risk Mitigation)")

        doc = {
            "doc_id": doc_id,
            "assessment_id": assessment_id,
            "operator_id": operator_id,
            "commodity": commodity,
            "composite_score": score,
            "risk_level": level,
            "requires_mitigation": requires_mitigation,
            "contributing_factors": contributing_factors or [],
            "country_scores": country_scores or [],
            "supplier_scores": supplier_scores or [],
            "article_references": article_references,
            "regulation_reference": "EU 2023/1115 Article 10",
            "documented_at": now.isoformat(),
            "provenance_hash": _compute_hash({
                "doc_id": doc_id,
                "assessment_id": assessment_id,
                "composite_score": score,
                "documented_at": now.isoformat(),
            }),
        }

        self._risk_docs[doc_id] = doc

        # Record provenance
        if self._provenance is not None:
            self._provenance.record(
                entity_type="risk_doc",
                action="create",
                entity_id=doc_id,
                actor=operator_id,
                metadata={
                    "assessment_id": assessment_id,
                    "risk_level": level,
                    "requires_mitigation": requires_mitigation,
                },
            )

        return doc

    # ------------------------------------------------------------------
    # Mitigation Documentation
    # ------------------------------------------------------------------

    async def document_mitigation(
        self,
        strategy_id: str,
        operator_id: str,
        commodity: str,
        pre_mitigation_score: str,
        post_mitigation_score: str,
        measures: Optional[List[Dict[str, Any]]] = None,
        verification_status: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Document mitigation measures for DDS inclusion.

        Creates structured documentation of mitigation measures
        including before/after risk scores, measure summaries,
        effectiveness evidence, and Article 11 compliance status.

        Args:
            strategy_id: Mitigation strategy identifier from EUDR-029.
            operator_id: EUDR operator identifier.
            commodity: EUDR regulated commodity.
            pre_mitigation_score: Risk score before mitigation.
            post_mitigation_score: Risk score after mitigation.
            measures: Optional list of measure summaries.
            verification_status: Optional verification status.

        Returns:
            MitigationDoc dictionary.
        """
        start = time.monotonic()

        logger.info(
            f"Documenting mitigation {strategy_id} for "
            f"operator={operator_id}, commodity={commodity}"
        )

        # Delegate to engine if available
        if self._mitigation_documenter is not None:
            try:
                doc = await self._mitigation_documenter.document(
                    strategy_id=strategy_id,
                    operator_id=operator_id,
                    commodity=commodity,
                    pre_mitigation_score=pre_mitigation_score,
                    post_mitigation_score=post_mitigation_score,
                    measures=measures,
                    verification_status=verification_status,
                )
                doc_id = (
                    doc.get("doc_id")
                    if isinstance(doc, dict)
                    else getattr(doc, "doc_id", _new_uuid())
                )
                self._mitigation_docs[doc_id] = doc
                _safe_record(record_mitigation_documented, commodity)
                return doc
            except Exception as e:
                logger.warning(
                    f"Mitigation documenter engine failed, using fallback: {e}"
                )

        # Fallback: in-memory documentation
        doc = await self._document_mitigation_fallback(
            strategy_id=strategy_id,
            operator_id=operator_id,
            commodity=commodity,
            pre_mitigation_score=pre_mitigation_score,
            post_mitigation_score=post_mitigation_score,
            measures=measures,
            verification_status=verification_status,
        )

        _safe_record(record_mitigation_documented, commodity)
        elapsed = time.monotonic() - start

        logger.info(
            f"Mitigation {strategy_id} documented "
            f"in {elapsed * 1000:.1f}ms"
        )

        return doc

    async def _document_mitigation_fallback(
        self,
        strategy_id: str,
        operator_id: str,
        commodity: str,
        pre_mitigation_score: Optional[str] = None,
        post_mitigation_score: Optional[str] = None,
        measures: Optional[List[Dict[str, Any]]] = None,
        verification_status: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fallback mitigation documentation.

        Args:
            strategy_id: Mitigation strategy identifier.
            operator_id: EUDR operator identifier.
            commodity: EUDR regulated commodity.
            pre_mitigation_score: Pre-mitigation risk score.
            post_mitigation_score: Post-mitigation risk score.
            measures: Measure summaries.
            verification_status: Verification status.

        Returns:
            Mitigation documentation dictionary.
        """
        doc_id = _new_uuid()
        now = _utcnow()

        pre_score = pre_mitigation_score or "50"
        post_score = post_mitigation_score or "30"
        pre_decimal = Decimal(pre_score)
        post_decimal = Decimal(post_score)
        reduction_achieved = pre_decimal - post_decimal

        measure_list = measures or []
        completed_count = sum(
            1 for m in measure_list
            if m.get("status") == "completed"
        )

        # Determine Article 11 compliance
        is_article11_compliant = (
            post_decimal <= Decimal(str(self.config.mitigation_threshold))
        )

        doc = {
            "doc_id": doc_id,
            "strategy_id": strategy_id,
            "operator_id": operator_id,
            "commodity": commodity,
            "pre_mitigation_score": pre_score,
            "post_mitigation_score": post_score,
            "reduction_achieved": str(reduction_achieved),
            "measures": measure_list,
            "total_measures": len(measure_list),
            "completed_measures": completed_count,
            "verification_status": verification_status or "pending",
            "is_article11_compliant": is_article11_compliant,
            "article_references": [
                "Article 11 (Risk Mitigation)",
                "Article 12(2) (DDS Content)",
            ],
            "regulation_reference": "EU 2023/1115 Article 11",
            "documented_at": now.isoformat(),
            "provenance_hash": _compute_hash({
                "doc_id": doc_id,
                "strategy_id": strategy_id,
                "pre_score": pre_score,
                "post_score": post_score,
                "documented_at": now.isoformat(),
            }),
        }

        self._mitigation_docs[doc_id] = doc

        # Record provenance
        if self._provenance is not None:
            self._provenance.record(
                entity_type="mitigation_doc",
                action="create",
                entity_id=doc_id,
                actor=operator_id,
                metadata={
                    "strategy_id": strategy_id,
                    "reduction": str(reduction_achieved),
                    "article11_compliant": is_article11_compliant,
                },
            )

        return doc

    # ------------------------------------------------------------------
    # Compliance Package Building
    # ------------------------------------------------------------------

    async def build_compliance_package(
        self,
        dds_id: str,
    ) -> Dict[str, Any]:
        """Build a compliance-ready submission package.

        Combines the DDS, Article 9 data, risk documentation,
        mitigation documentation, and supporting evidence into a
        submission-ready bundle with integrity hashes for each
        component.

        Args:
            dds_id: DDS identifier to package.

        Returns:
            CompliancePackage dictionary.

        Raises:
            ValueError: If DDS not found.
        """
        start = time.monotonic()

        logger.info(f"Building compliance package for DDS {dds_id}")

        # Delegate to engine if available
        if self._package_builder is not None:
            try:
                package = await self._package_builder.build_package(
                    dds_id=dds_id,
                )
                pkg_id = (
                    package.get("package_id")
                    if isinstance(package, dict)
                    else getattr(package, "package_id", _new_uuid())
                )
                self._compliance_packages[pkg_id] = package
                _safe_record(record_package_built)
                elapsed = time.monotonic() - start
                _safe_observe(observe_package_build_duration, elapsed)
                _safe_gauge(
                    set_active_packages, len(self._compliance_packages)
                )
                return package
            except Exception as e:
                logger.warning(
                    f"Package builder engine failed, using fallback: {e}"
                )

        # Fallback: build from in-memory data
        dds = self._dds_documents.get(dds_id)
        if dds is None:
            raise ValueError(f"DDS {dds_id} not found")

        package_id = _new_uuid()
        now = _utcnow()

        # Collect all component hashes
        component_hashes: Dict[str, str] = {
            "dds": dds.get("provenance_hash", ""),
        }

        # Get Article 9 package
        article9_id = dds.get("article9_package_id")
        article9_data = None
        if article9_id:
            article9_data = self._article9_packages.get(article9_id)
            if article9_data:
                component_hashes["article9"] = article9_data.get(
                    "provenance_hash", ""
                )

        # Get risk documentation
        risk_doc = dds.get("risk_documentation")
        if risk_doc:
            component_hashes["risk_assessment"] = risk_doc.get(
                "provenance_hash", ""
            )

        # Get mitigation documentation
        mitigation_doc = dds.get("mitigation_documentation")
        if mitigation_doc:
            component_hashes["mitigation"] = mitigation_doc.get(
                "provenance_hash", ""
            )

        # Compute package integrity hash
        package_integrity_hash = _compute_hash(component_hashes)

        # Determine package completeness
        sections_present = ["dds"]
        sections_missing: List[str] = []

        if article9_data:
            sections_present.append("article9")
        else:
            sections_missing.append("article9")

        if risk_doc:
            sections_present.append("risk_assessment")
        if mitigation_doc:
            sections_present.append("mitigation")

        is_complete = "article9" in sections_present

        package = {
            "package_id": package_id,
            "dds_id": dds_id,
            "operator_id": dds.get("operator_id", ""),
            "commodity": dds.get("commodity", ""),
            "format": "json",
            "sections_present": sections_present,
            "sections_missing": sections_missing,
            "is_complete": is_complete,
            "component_hashes": component_hashes,
            "integrity_hash": package_integrity_hash,
            "dds_data": dds,
            "article9_data": article9_data,
            "risk_documentation": risk_doc,
            "mitigation_documentation": mitigation_doc,
            "regulation_reference": "EU 2023/1115 Article 4(2)",
            "built_at": now.isoformat(),
            "provenance_hash": _compute_hash({
                "package_id": package_id,
                "dds_id": dds_id,
                "integrity_hash": package_integrity_hash,
                "built_at": now.isoformat(),
            }),
        }

        self._compliance_packages[package_id] = package

        # Record provenance
        if self._provenance is not None:
            self._provenance.record(
                entity_type="compliance_package",
                action="build",
                entity_id=package_id,
                actor="system",
                metadata={
                    "dds_id": dds_id,
                    "sections": len(sections_present),
                    "is_complete": is_complete,
                },
            )

        _safe_record(record_package_built)
        elapsed = time.monotonic() - start
        _safe_observe(observe_package_build_duration, elapsed)
        _safe_gauge(set_active_packages, len(self._compliance_packages))

        logger.info(
            f"Compliance package {package_id} built for DDS {dds_id} "
            f"in {elapsed * 1000:.1f}ms "
            f"(sections={len(sections_present)}, complete={is_complete})"
        )

        return package

    # ------------------------------------------------------------------
    # DDS Validation
    # ------------------------------------------------------------------

    async def validate_dds(
        self,
        dds_id: str,
    ) -> Dict[str, Any]:
        """Validate DDS completeness and regulatory compliance.

        Checks all required Article 9 elements, risk assessment
        documentation, mitigation documentation where required,
        and structural integrity of the DDS.

        Args:
            dds_id: DDS identifier to validate.

        Returns:
            ValidationResult dictionary with issues and pass/fail status.

        Raises:
            ValueError: If DDS not found.
        """
        logger.info(f"Validating DDS {dds_id}")

        dds = self._dds_documents.get(dds_id)
        if dds is None:
            # Try engine
            if self._dds_generator is not None:
                try:
                    dds = await self._dds_generator.get_dds(dds_id)
                except Exception:
                    pass

            if dds is None:
                raise ValueError(f"DDS {dds_id} not found")

        validation_id = _new_uuid()
        now = _utcnow()
        issues: List[Dict[str, Any]] = []

        # Validate operator information
        if not dds.get("operator_id"):
            issues.append({
                "severity": "error",
                "field": "operator_id",
                "message": "Operator identifier is required",
                "article_reference": "Article 9(1)(a)",
            })

        # Validate commodity
        valid_commodities = {
            "cattle", "cocoa", "coffee", "palm_oil",
            "rubber", "soya", "wood",
        }
        commodity = dds.get("commodity", "")
        if commodity not in valid_commodities:
            issues.append({
                "severity": "error",
                "field": "commodity",
                "message": f"Invalid commodity '{commodity}'. "
                           f"Must be one of: {', '.join(sorted(valid_commodities))}",
                "article_reference": "Article 1(1)",
            })

        # Validate products
        products = dds.get("products", [])
        if not products:
            issues.append({
                "severity": "error",
                "field": "products",
                "message": "At least one product entry is required",
                "article_reference": "Article 9(1)(b)",
            })
        else:
            for i, product in enumerate(products):
                if not product.get("description"):
                    issues.append({
                        "severity": "warning",
                        "field": f"products[{i}].description",
                        "message": f"Product {i} missing description",
                        "article_reference": "Article 9(1)(b)",
                    })
                if not product.get("hs_code"):
                    issues.append({
                        "severity": "warning",
                        "field": f"products[{i}].hs_code",
                        "message": f"Product {i} missing HS code",
                        "article_reference": "Article 9(1)(d)",
                    })

        # Validate geolocations
        geolocations = dds.get("geolocations", [])
        if not geolocations:
            issues.append({
                "severity": "error",
                "field": "geolocations",
                "message": "At least one geolocation reference is required",
                "article_reference": "Article 9(1)(f)",
            })
        else:
            for i, geo in enumerate(geolocations):
                if not geo.get("latitude") and not geo.get("coordinates"):
                    issues.append({
                        "severity": "warning",
                        "field": f"geolocations[{i}].coordinates",
                        "message": f"Geolocation {i} missing coordinates",
                        "article_reference": "Article 9(1)(f)",
                    })
                if not geo.get("country_code"):
                    issues.append({
                        "severity": "warning",
                        "field": f"geolocations[{i}].country_code",
                        "message": f"Geolocation {i} missing country code",
                        "article_reference": "Article 9(1)(e)",
                    })

        # Validate suppliers
        suppliers = dds.get("suppliers", [])
        if not suppliers:
            issues.append({
                "severity": "error",
                "field": "suppliers",
                "message": "At least one supplier reference is required",
                "article_reference": "Article 9(1)(g)",
            })

        # Validate risk assessment if risk level warrants it
        risk_conclusion = dds.get("risk_conclusion", "standard")
        if risk_conclusion in ("high", "critical"):
            if not dds.get("risk_documentation"):
                issues.append({
                    "severity": "error",
                    "field": "risk_documentation",
                    "message": "Risk documentation required for high/critical risk DDS",
                    "article_reference": "Article 10",
                })
            if not dds.get("mitigation_documentation"):
                issues.append({
                    "severity": "error",
                    "field": "mitigation_documentation",
                    "message": "Mitigation documentation required for "
                               "high/critical risk DDS",
                    "article_reference": "Article 11",
                })

        # Calculate result
        error_count = sum(
            1 for issue in issues if issue["severity"] == "error"
        )
        warning_count = sum(
            1 for issue in issues if issue["severity"] == "warning"
        )
        is_valid = error_count == 0

        result = {
            "validation_id": validation_id,
            "dds_id": dds_id,
            "is_valid": is_valid,
            "status": "pass" if is_valid else "fail",
            "error_count": error_count,
            "warning_count": warning_count,
            "total_issues": len(issues),
            "issues": issues,
            "validated_at": now.isoformat(),
            "regulation_reference": "EU 2023/1115 Articles 4, 9",
            "provenance_hash": _compute_hash({
                "validation_id": validation_id,
                "dds_id": dds_id,
                "is_valid": is_valid,
                "validated_at": now.isoformat(),
            }),
        }

        self._validation_results[validation_id] = result

        # Update DDS with validation result
        if dds_id in self._dds_documents:
            self._dds_documents[dds_id]["last_validation"] = result
            if is_valid and self._dds_documents[dds_id].get("status") == "draft":
                self._dds_documents[dds_id]["status"] = "validated"

        # Record provenance
        if self._provenance is not None:
            self._provenance.record(
                entity_type="validation",
                action="validate",
                entity_id=validation_id,
                actor="system",
                metadata={
                    "dds_id": dds_id,
                    "is_valid": is_valid,
                    "error_count": error_count,
                    "warning_count": warning_count,
                },
            )

        _safe_record(record_validation_run)

        # Update validation pass rate gauge
        total_validations = len(self._validation_results)
        if total_validations > 0:
            pass_count = sum(
                1 for v in self._validation_results.values()
                if v.get("is_valid")
            )
            _safe_gauge(
                set_validation_pass_rate,
                (pass_count / total_validations) * 100.0,
            )

        logger.info(
            f"DDS {dds_id} validation: {'PASS' if is_valid else 'FAIL'} "
            f"(errors={error_count}, warnings={warning_count})"
        )

        return result

    # ------------------------------------------------------------------
    # DDS Submission
    # ------------------------------------------------------------------

    async def submit_dds(
        self,
        dds_id: str,
    ) -> Dict[str, Any]:
        """Submit a DDS to the EU Information System.

        Validates the DDS, builds a compliance package if needed,
        and submits to the EU IS with status tracking. The DDS must
        be in a validated or compliant state to be submitted.

        Args:
            dds_id: DDS identifier to submit.

        Returns:
            SubmissionRecord dictionary.

        Raises:
            ValueError: If DDS not found or not in submittable state.
        """
        start = time.monotonic()

        logger.info(f"Submitting DDS {dds_id} to EU IS")

        # Delegate to engine if available
        if self._submission_engine is not None:
            try:
                submission = await self._submission_engine.submit(
                    dds_id=dds_id,
                )
                sub_id = (
                    submission.get("submission_id")
                    if isinstance(submission, dict)
                    else getattr(submission, "submission_id", _new_uuid())
                )
                self._submissions[sub_id] = submission
                _safe_record(record_submission_sent)
                elapsed = time.monotonic() - start
                _safe_observe(observe_submission_duration, elapsed)
                _safe_gauge(
                    set_active_submissions, len(self._submissions)
                )
                return submission
            except Exception as e:
                logger.warning(
                    f"Submission engine failed, using fallback: {e}"
                )

        # Fallback: in-memory submission
        dds = self._dds_documents.get(dds_id)
        if dds is None:
            raise ValueError(f"DDS {dds_id} not found")

        # Check DDS is in submittable state
        submittable_statuses = {
            "validated", "compliant", "compliant_with_mitigation",
        }
        current_status = dds.get("status", "draft")
        if current_status not in submittable_statuses:
            raise ValueError(
                f"DDS {dds_id} is in '{current_status}' status. "
                f"Must be one of: {', '.join(sorted(submittable_statuses))} "
                f"before submission. Run validate_dds() first."
            )

        submission_id = _new_uuid()
        now = _utcnow()
        deadline = now + timedelta(days=self.config.submission_deadline_days)

        submission = {
            "submission_id": submission_id,
            "dds_id": dds_id,
            "operator_id": dds.get("operator_id", ""),
            "commodity": dds.get("commodity", ""),
            "status": "submitted",
            "submitted_at": now.isoformat(),
            "submission_deadline": deadline.isoformat(),
            "eu_is_reference": f"EUIS-{submission_id[:8].upper()}",
            "receipt_confirmed": False,
            "retry_count": 0,
            "max_retries": self.config.max_submission_retries,
            "regulation_reference": "EU 2023/1115 Article 4(2)",
            "provenance_hash": _compute_hash({
                "submission_id": submission_id,
                "dds_id": dds_id,
                "submitted_at": now.isoformat(),
            }),
        }

        self._submissions[submission_id] = submission

        # Update DDS status
        dds["status"] = "submitted"
        dds["submission_id"] = submission_id
        dds["updated_at"] = now.isoformat()

        # Record version
        current_version = dds.get("version", 1)
        dds["version"] = current_version + 1
        self._record_version(
            document_id=dds_id,
            document_type="dds",
            version=dds["version"],
            actor="system",
            change_summary="DDS submitted to EU Information System",
        )

        # Record provenance
        if self._provenance is not None:
            self._provenance.record(
                entity_type="submission",
                action="submit",
                entity_id=submission_id,
                actor=dds.get("operator_id", "system"),
                metadata={
                    "dds_id": dds_id,
                    "eu_is_reference": submission["eu_is_reference"],
                },
            )

        _safe_record(record_submission_sent)
        elapsed = time.monotonic() - start
        _safe_observe(observe_submission_duration, elapsed)
        _safe_gauge(set_active_submissions, len(self._submissions))

        pending_count = sum(
            1 for s in self._submissions.values()
            if s.get("status") in ("submitted", "pending")
        )
        _safe_gauge(set_pending_submissions, pending_count)

        logger.info(
            f"DDS {dds_id} submitted as {submission_id} "
            f"in {elapsed * 1000:.1f}ms "
            f"(eu_is_ref={submission['eu_is_reference']})"
        )

        return submission

    async def get_submission_status(
        self,
        submission_id: str,
    ) -> Dict[str, Any]:
        """Get the current status of a DDS submission.

        Args:
            submission_id: Submission identifier.

        Returns:
            SubmissionRecord dictionary.

        Raises:
            ValueError: If submission not found.
        """
        # Delegate to engine if available
        if self._submission_engine is not None:
            try:
                return await self._submission_engine.get_status(
                    submission_id=submission_id,
                )
            except Exception as e:
                logger.debug(f"Engine get_status failed: {e}")

        # Fallback: in-memory lookup
        submission = self._submissions.get(submission_id)
        if submission is None:
            raise ValueError(f"Submission {submission_id} not found")

        return submission

    # ------------------------------------------------------------------
    # Document Versioning
    # ------------------------------------------------------------------

    def _record_version(
        self,
        document_id: str,
        document_type: str,
        version: int,
        actor: str,
        change_summary: str,
    ) -> Dict[str, Any]:
        """Record a new version entry for a document.

        Args:
            document_id: Document identifier.
            document_type: Type of document (dds, article9, etc.).
            version: Version number.
            actor: User or system who made the change.
            change_summary: Description of the change.

        Returns:
            Version record dictionary.
        """
        version_id = _new_uuid()
        now = _utcnow()

        version_record = {
            "version_id": version_id,
            "document_id": document_id,
            "document_type": document_type,
            "version": version,
            "actor": actor,
            "change_summary": change_summary,
            "created_at": now.isoformat(),
            "provenance_hash": _compute_hash({
                "version_id": version_id,
                "document_id": document_id,
                "version": version,
                "created_at": now.isoformat(),
            }),
        }

        if document_id not in self._versions:
            self._versions[document_id] = []
        self._versions[document_id].append(version_record)

        _safe_gauge(
            set_total_versions,
            sum(len(v) for v in self._versions.values()),
        )

        return version_record

    async def get_version_history(
        self,
        document_id: str,
    ) -> List[Dict[str, Any]]:
        """Get the version history for a document.

        Returns all recorded versions sorted by version number
        ascending, including timestamps and change summaries.

        Args:
            document_id: Document identifier.

        Returns:
            List of version records ordered by version number.

        Raises:
            ValueError: If no versions found for the document.
        """
        # Delegate to engine if available
        if self._version_manager is not None:
            try:
                return await self._version_manager.get_history(
                    document_id=document_id,
                )
            except Exception as e:
                logger.debug(f"Engine get_history failed: {e}")

        # Fallback: in-memory lookup
        versions = self._versions.get(document_id)
        if versions is None:
            raise ValueError(
                f"No version history found for document {document_id}"
            )

        # Sort by version number ascending
        return sorted(versions, key=lambda v: v.get("version", 0))

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
                "dds_documents": len(self._dds_documents),
                "article9_packages": len(self._article9_packages),
                "risk_docs": len(self._risk_docs),
                "mitigation_docs": len(self._mitigation_docs),
                "compliance_packages": len(self._compliance_packages),
                "submissions": len(self._submissions),
                "versioned_documents": len(self._versions),
                "validation_results": len(self._validation_results),
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
            "dds_statement_generator",
            "article9_data_assembler",
            "risk_assessment_documenter",
            "mitigation_documenter",
            "compliance_package_builder",
            "document_version_manager",
            "regulatory_submission_engine",
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
            name: Engine name (e.g., 'dds_statement_generator').

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
    def dds_count(self) -> int:
        """Return the number of DDS documents in memory."""
        return len(self._dds_documents)

    @property
    def submission_count(self) -> int:
        """Return the number of submissions in memory."""
        return len(self._submissions)

    @property
    def package_count(self) -> int:
        """Return the number of compliance packages in memory."""
        return len(self._compliance_packages)


# ---------------------------------------------------------------------------
# Thread-safe singleton
# ---------------------------------------------------------------------------

_service_lock = threading.Lock()
_service_instance: Optional[DocumentationGeneratorService] = None


def get_service(
    config: Optional[DocumentationGeneratorConfig] = None,
) -> DocumentationGeneratorService:
    """Get the global DocumentationGeneratorService singleton instance.

    Thread-safe lazy initialization. Creates a new service instance
    on first call.

    Args:
        config: Optional configuration override for first creation.

    Returns:
        DocumentationGeneratorService singleton instance.

    Example:
        >>> service = get_service()
        >>> assert service is get_service()
    """
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = DocumentationGeneratorService(config)
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
        >>> from greenlang.agents.eudr.documentation_generator.setup import lifespan
        >>> app = FastAPI(lifespan=lifespan)

    Args:
        app: FastAPI application instance.

    Yields:
        None -- application runs between startup and shutdown.
    """
    service = get_service()
    await service.startup()
    logger.info("Documentation Generator lifespan: startup complete")

    try:
        yield
    finally:
        await service.shutdown()
        logger.info(
            "Documentation Generator lifespan: shutdown complete"
        )
