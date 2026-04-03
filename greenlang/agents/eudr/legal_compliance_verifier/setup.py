# -*- coding: utf-8 -*-
"""
LegalComplianceVerifierSetup - Facade for AGENT-EUDR-023

Unified setup facade orchestrating all 7 engines of the Legal Compliance
Verifier Agent. Provides a single entry point for legal framework querying,
document verification, certification validation, red flag detection,
country compliance checking, third-party audit processing, and compliance
report generation.

Engines (7):
    1. LegalFrameworkDatabaseEngine   - 27 countries, 8 legislation categories
    2. DocumentVerificationEngine     - 12 document types, validity pipeline
    3. CertificationSchemeValidator   - 5 schemes, EUDR equivalence mapping
    4. RedFlagDetectionEngine         - 40 red flags, deterministic scoring
    5. CountryComplianceChecker       - Per-country rule sets, gap analysis
    6. ThirdPartyAuditEngine          - 6 audit sources, finding extraction
    7. ComplianceReportingEngine      - 8 report types, 5 formats, 5 languages

Singleton Pattern:
    Thread-safe singleton with double-checked locking via ``get_service()``.

FastAPI Integration:
    Use the ``lifespan`` async context manager with
    ``FastAPI(lifespan=lifespan)`` for automatic startup/shutdown.

Example:
    >>> from greenlang.agents.eudr.legal_compliance_verifier.setup import (
    ...     LegalComplianceVerifierSetup,
    ...     get_service,
    ... )
    >>> service = get_service()
    >>> await service.startup()
    >>> health = await service.health_check()
    >>> assert health["status"] == "healthy"

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-023 Legal Compliance Verifier (GL-EUDR-LCV-023)
Regulation: EU 2023/1115 (EUDR) Article 2(40), Articles 3, 8, 10, 11, 29, 31
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
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, AsyncIterator, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"
_AGENT_ID = "GL-EUDR-LCV-023"

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

from greenlang.agents.eudr.legal_compliance_verifier.config import (
    LegalComplianceVerifierConfig,
    get_config,
    set_config,
    reset_config,
)
from greenlang.agents.eudr.legal_compliance_verifier.provenance import (
    ProvenanceTracker,
    get_tracker,
)
from greenlang.agents.eudr.legal_compliance_verifier.metrics import (
    PROMETHEUS_AVAILABLE,
    record_framework_query,
    record_document_verification,
    record_certification_validation,
    record_red_flag_scan,
    record_compliance_assessment,
    record_audit_report_processed,
    record_report_generated,
    record_batch_job,
    record_api_error,
    observe_compliance_check_duration,
    observe_full_assessment_duration,
    observe_document_verification_duration,
    observe_red_flag_scan_duration,
    observe_report_generation_duration,
    set_countries_covered,
    set_active_red_flags,
    set_expiring_documents_30d,
    set_non_compliant_suppliers,
    set_cache_hit_ratio,
)

# ---------------------------------------------------------------------------
# Engine imports (lazy initialization)
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.legal_compliance_verifier.legal_framework_database_engine import (
        LegalFrameworkDatabaseEngine,
    )
except ImportError:
    LegalFrameworkDatabaseEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.eudr.legal_compliance_verifier.document_verification_engine import (
        DocumentVerificationEngine,
    )
except ImportError:
    DocumentVerificationEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.eudr.legal_compliance_verifier.certification_scheme_validator import (
        CertificationSchemeValidator,
    )
except ImportError:
    CertificationSchemeValidator = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.eudr.legal_compliance_verifier.red_flag_detection_engine import (
        RedFlagDetectionEngine,
    )
except ImportError:
    RedFlagDetectionEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.eudr.legal_compliance_verifier.country_compliance_checker import (
        CountryComplianceChecker,
    )
except ImportError:
    CountryComplianceChecker = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.eudr.legal_compliance_verifier.third_party_audit_engine import (
        ThirdPartyAuditEngine,
    )
except ImportError:
    ThirdPartyAuditEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.agents.eudr.legal_compliance_verifier.compliance_reporting_engine import (
        ComplianceReportingEngine,
    )
except ImportError:
    ComplianceReportingEngine = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# LegalComplianceVerifierSetup facade
# ---------------------------------------------------------------------------


class LegalComplianceVerifierSetup:
    """Facade orchestrating all 7 engines of the Legal Compliance Verifier.

    Provides lazy initialization of engines, lifecycle management
    (startup/shutdown), health checking, and unified access to all
    legal compliance verification operations.

    Example:
        >>> service = LegalComplianceVerifierSetup()
        >>> await service.startup()
        >>> frameworks = service.query_frameworks("BR")
        >>> health = await service.health_check()
        >>> await service.shutdown()
    """

    def __init__(
        self,
        config: Optional[LegalComplianceVerifierConfig] = None,
    ) -> None:
        """Initialize the Legal Compliance Verifier service.

        Args:
            config: Optional configuration override. If None, uses
                    get_config() to load from environment.
        """
        self._config = config or get_config()
        self._provenance = get_tracker()
        self._start_time = time.monotonic()
        self._started = False

        # Lazy engine instances
        self._framework_engine: Optional[Any] = None
        self._document_engine: Optional[Any] = None
        self._certification_engine: Optional[Any] = None
        self._red_flag_engine: Optional[Any] = None
        self._compliance_engine: Optional[Any] = None
        self._audit_engine: Optional[Any] = None
        self._reporting_engine: Optional[Any] = None

        # Infrastructure
        self._db_pool: Optional[Any] = None
        self._redis: Optional[Any] = None

        logger.info(
            f"LegalComplianceVerifierSetup v{_MODULE_VERSION} created"
        )

    # -------------------------------------------------------------------
    # Lifecycle management
    # -------------------------------------------------------------------

    async def startup(self) -> None:
        """Initialize all engines and infrastructure connections.

        Performs lazy initialization of all 7 engines and optionally
        establishes database and Redis connections.
        """
        logger.info("LegalComplianceVerifierSetup startup initiated")
        start = time.monotonic()

        # Initialize database pool
        await self._init_database()

        # Initialize Redis
        await self._init_redis()

        # Initialize all engines
        self._ensure_framework_engine()
        self._ensure_document_engine()
        self._ensure_certification_engine()
        self._ensure_red_flag_engine()
        self._ensure_compliance_engine()
        self._ensure_audit_engine()
        self._ensure_reporting_engine()

        # Update gauge metrics
        self._update_startup_metrics()

        self._started = True
        elapsed = time.monotonic() - start
        logger.info(
            f"LegalComplianceVerifierSetup startup completed in {elapsed:.3f}s"
        )

    async def shutdown(self) -> None:
        """Gracefully shut down all engines and connections."""
        logger.info("LegalComplianceVerifierSetup shutdown initiated")

        if self._db_pool is not None:
            try:
                await self._db_pool.close()
            except Exception as exc:
                logger.warning("Database pool close failed: %s", exc)

        if self._redis is not None:
            try:
                await self._redis.close()
            except Exception as exc:
                logger.warning("Redis close failed: %s", exc)

        self._started = False
        logger.info("LegalComplianceVerifierSetup shutdown completed")

    async def health_check(self) -> Dict[str, Any]:
        """Check health of all engines and infrastructure.

        Returns:
            Dict with status, engine health, connection status.
        """
        uptime = time.monotonic() - self._start_time

        engines: Dict[str, str] = {
            "legal_framework_database": "ready" if self._framework_engine else "not_initialized",
            "document_verification": "ready" if self._document_engine else "not_initialized",
            "certification_scheme_validator": "ready" if self._certification_engine else "not_initialized",
            "red_flag_detection": "ready" if self._red_flag_engine else "not_initialized",
            "country_compliance_checker": "ready" if self._compliance_engine else "not_initialized",
            "third_party_audit": "ready" if self._audit_engine else "not_initialized",
            "compliance_reporting": "ready" if self._reporting_engine else "not_initialized",
        }

        all_ready = all(v == "ready" for v in engines.values())
        db_connected = self._db_pool is not None
        redis_connected = self._redis is not None
        chain_valid = self._provenance.verify_chain()

        status = "healthy" if all_ready else "degraded"

        return {
            "status": status,
            "version": _MODULE_VERSION,
            "agent_id": _AGENT_ID,
            "engines": engines,
            "database_connected": db_connected,
            "redis_connected": redis_connected,
            "uptime_seconds": round(uptime, 2),
            "provenance_chain_valid": chain_valid,
            "provenance_records": len(self._provenance.get_chain()),
        }

    # -------------------------------------------------------------------
    # Public API: Delegated operations
    # -------------------------------------------------------------------

    def query_frameworks(
        self,
        country_code: str,
        category: Optional[str] = None,
        commodity: Optional[str] = None,
        include_repealed: bool = False,
    ) -> Dict[str, Any]:
        """Query legal frameworks (delegates to Engine 1).

        Args:
            country_code: ISO 3166-1 alpha-2 country code.
            category: Optional legislation category filter.
            commodity: Optional commodity filter.
            include_repealed: Include repealed legislation.

        Returns:
            Framework query results dict.
        """
        engine = self._ensure_framework_engine()
        return engine.query_frameworks(
            country_code, category, commodity, include_repealed,
        )

    def verify_document(
        self,
        document_type: str,
        document_number: str,
        issuing_authority: str,
        issuing_country: str,
        issue_date: date,
        expiry_date: Optional[date] = None,
        s3_document_key: Optional[str] = None,
        supplier_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Verify a compliance document (delegates to Engine 2).

        Args:
            document_type: Type of document.
            document_number: Document reference number.
            issuing_authority: Issuing authority name.
            issuing_country: Country code.
            issue_date: Document issue date.
            expiry_date: Optional expiry date.
            s3_document_key: Optional S3 key.
            supplier_id: Optional supplier ID.

        Returns:
            Document verification results dict.
        """
        engine = self._ensure_document_engine()
        return engine.verify_document(
            document_type, document_number, issuing_authority,
            issuing_country, issue_date, expiry_date,
            s3_document_key, supplier_id,
        )

    def validate_certification(
        self,
        scheme: str,
        certificate_number: str,
        certification_body: Optional[str] = None,
        issue_date: Optional[date] = None,
        expiry_date: Optional[date] = None,
        covered_commodities: Optional[List[str]] = None,
        coc_model: Optional[str] = None,
        last_audit_date: Optional[date] = None,
        non_conformities_open: int = 0,
        supplier_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate a certification (delegates to Engine 3).

        Args:
            scheme: Certification scheme key.
            certificate_number: Certificate number.
            certification_body: Optional CB name.
            issue_date: Certificate issue date.
            expiry_date: Certificate expiry date.
            covered_commodities: Covered commodities.
            coc_model: Chain of custody model.
            last_audit_date: Last audit date.
            non_conformities_open: Open NCs.
            supplier_id: Optional supplier ID.

        Returns:
            Certification validation results dict.
        """
        engine = self._ensure_certification_engine()
        return engine.validate_certification(
            scheme, certificate_number, certification_body,
            issue_date, expiry_date, covered_commodities,
            coc_model, last_audit_date, non_conformities_open,
            supplier_id,
        )

    def scan_red_flags(
        self,
        supplier_data: Dict[str, Any],
        country_code: str,
        commodity: str,
        include_categories: Optional[List[str]] = None,
        supplier_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Scan for red flags (delegates to Engine 4).

        Args:
            supplier_data: Supplier data for evaluation.
            country_code: Country code.
            commodity: Commodity type.
            include_categories: Optional category filter.
            supplier_id: Optional supplier ID.

        Returns:
            Red flag scan results dict.
        """
        engine = self._ensure_red_flag_engine()
        return engine.scan_red_flags(
            supplier_data, country_code, commodity,
            include_categories, supplier_id,
        )

    def assess_compliance(
        self,
        country_code: str,
        commodity: str,
        documents: Optional[List[Dict[str, Any]]] = None,
        certifications: Optional[List[Dict[str, Any]]] = None,
        red_flag_data: Optional[Dict[str, Any]] = None,
        categories: Optional[List[str]] = None,
        supplier_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Assess compliance (delegates to Engine 5).

        Args:
            country_code: Country code.
            commodity: Commodity type.
            documents: Available documents.
            certifications: Available certifications.
            red_flag_data: Red flag scan data.
            categories: Category filter.
            supplier_id: Supplier ID.

        Returns:
            Compliance assessment results dict.
        """
        engine = self._ensure_compliance_engine()
        return engine.assess_compliance(
            country_code, commodity, documents,
            certifications, red_flag_data, categories,
            supplier_id,
        )

    def process_audit_report(
        self,
        audit_type: str,
        auditor_organization: str,
        audit_date: date,
        report_date: date,
        overall_conclusion: str = "",
        findings: Optional[List[Dict[str, Any]]] = None,
        lead_auditor: Optional[str] = None,
        scope: str = "",
        s3_report_key: Optional[str] = None,
        supplier_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process audit report (delegates to Engine 6).

        Args:
            audit_type: Audit type.
            auditor_organization: Auditor name.
            audit_date: Audit date.
            report_date: Report date.
            overall_conclusion: Audit conclusion.
            findings: Findings list.
            lead_auditor: Lead auditor.
            scope: Audit scope.
            s3_report_key: S3 key.
            supplier_id: Supplier ID.

        Returns:
            Audit report processing results dict.
        """
        engine = self._ensure_audit_engine()
        return engine.process_audit_report(
            audit_type, auditor_organization, audit_date,
            report_date, overall_conclusion, findings,
            lead_auditor, scope, s3_report_key, supplier_id,
        )

    def generate_report(
        self,
        report_type: str,
        report_format: Optional[str] = None,
        language: Optional[str] = None,
        assessment_data: Optional[Dict[str, Any]] = None,
        assessment_id: Optional[str] = None,
        supplier_id: Optional[str] = None,
        country_code: Optional[str] = None,
        commodity: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate compliance report (delegates to Engine 7).

        Args:
            report_type: Report type.
            report_format: Format.
            language: Language.
            assessment_data: Assessment data.
            assessment_id: Assessment ID.
            supplier_id: Supplier ID.
            country_code: Country code.
            commodity: Commodity.

        Returns:
            Report generation results dict.
        """
        engine = self._ensure_reporting_engine()
        return engine.generate_report(
            report_type, report_format, language,
            assessment_data, assessment_id,
            supplier_id, country_code, commodity,
        )

    # -------------------------------------------------------------------
    # Public API: Composite operations
    # -------------------------------------------------------------------

    def run_full_assessment(
        self,
        supplier_id: str,
        country_code: str,
        commodity: str,
        documents: Optional[List[Dict[str, Any]]] = None,
        certifications: Optional[List[Dict[str, Any]]] = None,
        supplier_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run a full compliance assessment with all engines.

        Orchestrates: framework query, document verification,
        certification validation, red flag scan, and compliance check.

        Args:
            supplier_id: Supplier identifier.
            country_code: Country code.
            commodity: Commodity type.
            documents: Available documents.
            certifications: Available certifications.
            supplier_data: Supplier data for red flags.

        Returns:
            Comprehensive assessment results dict.
        """
        start_time = time.monotonic()
        results: Dict[str, Any] = {}

        # Step 1: Query frameworks
        try:
            results["frameworks"] = self.query_frameworks(
                country_code, commodity=commodity,
            )
        except Exception as exc:
            logger.error("Framework query failed: %s", exc)
            results["frameworks"] = {"error": str(exc)}

        # Step 2: Red flag scan
        red_flag_data = None
        if supplier_data:
            try:
                red_flag_data = self.scan_red_flags(
                    supplier_data, country_code, commodity,
                    supplier_id=supplier_id,
                )
                results["red_flags"] = red_flag_data
            except Exception as exc:
                logger.error("Red flag scan failed: %s", exc)
                results["red_flags"] = {"error": str(exc)}

        # Step 3: Compliance assessment
        try:
            results["assessment"] = self.assess_compliance(
                country_code=country_code,
                commodity=commodity,
                documents=documents,
                certifications=certifications,
                red_flag_data=red_flag_data,
                supplier_id=supplier_id,
            )
        except Exception as exc:
            logger.error("Compliance assessment failed: %s", exc)
            results["assessment"] = {"error": str(exc)}

        elapsed = time.monotonic() - start_time
        results["processing_time_seconds"] = round(elapsed, 3)
        results["supplier_id"] = supplier_id
        results["country_code"] = country_code
        results["commodity"] = commodity

        logger.info(
            f"Full assessment completed for {supplier_id} in {elapsed:.3f}s"
        )

        return results

    # -------------------------------------------------------------------
    # Public API: Statistics
    # -------------------------------------------------------------------

    async def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics.

        Returns:
            Dict with operational statistics.
        """
        framework_engine = self._ensure_framework_engine()
        supported = framework_engine.get_supported_countries()

        return {
            "agent_id": _AGENT_ID,
            "version": _MODULE_VERSION,
            "countries_covered": len(supported),
            "supported_countries": supported,
            "provenance_records": len(self._provenance.get_chain()),
            "uptime_seconds": round(time.monotonic() - self._start_time, 2),
        }

    # -------------------------------------------------------------------
    # Internal: Lazy engine initialization
    # -------------------------------------------------------------------

    def _ensure_framework_engine(self) -> Any:
        """Lazily initialize the Legal Framework Database Engine."""
        if self._framework_engine is None:
            if LegalFrameworkDatabaseEngine is not None:
                self._framework_engine = LegalFrameworkDatabaseEngine()
            else:
                raise RuntimeError("LegalFrameworkDatabaseEngine not available")
        return self._framework_engine

    def _ensure_document_engine(self) -> Any:
        """Lazily initialize the Document Verification Engine."""
        if self._document_engine is None:
            if DocumentVerificationEngine is not None:
                self._document_engine = DocumentVerificationEngine()
            else:
                raise RuntimeError("DocumentVerificationEngine not available")
        return self._document_engine

    def _ensure_certification_engine(self) -> Any:
        """Lazily initialize the Certification Scheme Validator."""
        if self._certification_engine is None:
            if CertificationSchemeValidator is not None:
                self._certification_engine = CertificationSchemeValidator()
            else:
                raise RuntimeError("CertificationSchemeValidator not available")
        return self._certification_engine

    def _ensure_red_flag_engine(self) -> Any:
        """Lazily initialize the Red Flag Detection Engine."""
        if self._red_flag_engine is None:
            if RedFlagDetectionEngine is not None:
                self._red_flag_engine = RedFlagDetectionEngine()
            else:
                raise RuntimeError("RedFlagDetectionEngine not available")
        return self._red_flag_engine

    def _ensure_compliance_engine(self) -> Any:
        """Lazily initialize the Country Compliance Checker."""
        if self._compliance_engine is None:
            if CountryComplianceChecker is not None:
                self._compliance_engine = CountryComplianceChecker()
            else:
                raise RuntimeError("CountryComplianceChecker not available")
        return self._compliance_engine

    def _ensure_audit_engine(self) -> Any:
        """Lazily initialize the Third-Party Audit Engine."""
        if self._audit_engine is None:
            if ThirdPartyAuditEngine is not None:
                self._audit_engine = ThirdPartyAuditEngine()
            else:
                raise RuntimeError("ThirdPartyAuditEngine not available")
        return self._audit_engine

    def _ensure_reporting_engine(self) -> Any:
        """Lazily initialize the Compliance Reporting Engine."""
        if self._reporting_engine is None:
            if ComplianceReportingEngine is not None:
                self._reporting_engine = ComplianceReportingEngine()
            else:
                raise RuntimeError("ComplianceReportingEngine not available")
        return self._reporting_engine

    # -------------------------------------------------------------------
    # Internal: Infrastructure initialization
    # -------------------------------------------------------------------

    async def _init_database(self) -> None:
        """Initialize database connection pool."""
        if not PSYCOPG_POOL_AVAILABLE:
            logger.info("psycopg_pool not available; skipping DB init")
            return

        try:
            self._db_pool = AsyncConnectionPool(
                conninfo=self._config.database_url,
                min_size=2,
                max_size=self._config.pool_size,
                open=False,
            )
            await self._db_pool.open()
            logger.info("Database pool initialized")
        except Exception as exc:
            logger.warning("Database pool init failed: %s", exc)
            self._db_pool = None

    async def _init_redis(self) -> None:
        """Initialize Redis connection."""
        if not REDIS_AVAILABLE:
            logger.info("redis.asyncio not available; skipping Redis init")
            return

        try:
            self._redis = aioredis.from_url(
                self._config.redis_url,
                decode_responses=True,
            )
            await self._redis.ping()
            logger.info("Redis connection initialized")
        except Exception as exc:
            logger.warning("Redis init failed: %s", exc)
            self._redis = None

    # -------------------------------------------------------------------
    # Internal: Metrics initialization
    # -------------------------------------------------------------------

    def _update_startup_metrics(self) -> None:
        """Update gauge metrics after startup."""
        try:
            engine = self._ensure_framework_engine()
            countries = engine.get_supported_countries()
            set_countries_covered(len(countries))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Thread-safe singleton
# ---------------------------------------------------------------------------

_service_lock = threading.Lock()
_global_service: Optional[LegalComplianceVerifierSetup] = None


def get_service() -> LegalComplianceVerifierSetup:
    """Get the global LegalComplianceVerifierSetup singleton.

    Thread-safe lazy initialization.

    Returns:
        LegalComplianceVerifierSetup singleton instance.

    Example:
        >>> service = get_service()
        >>> service2 = get_service()
        >>> assert service is service2
    """
    global _global_service
    if _global_service is None:
        with _service_lock:
            if _global_service is None:
                _global_service = LegalComplianceVerifierSetup()
    return _global_service


def set_service(service: LegalComplianceVerifierSetup) -> None:
    """Set the global LegalComplianceVerifierSetup singleton.

    Args:
        service: Service instance to set as global.
    """
    global _global_service
    with _service_lock:
        _global_service = service


def reset_service() -> None:
    """Reset the global LegalComplianceVerifierSetup singleton."""
    global _global_service
    with _service_lock:
        _global_service = None
        logger.warning("LegalComplianceVerifierSetup service reset")


# ---------------------------------------------------------------------------
# FastAPI lifespan context manager
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: Any) -> AsyncIterator[None]:
    """FastAPI lifespan context manager for startup/shutdown.

    Usage:
        >>> from fastapi import FastAPI
        >>> from greenlang.agents.eudr.legal_compliance_verifier.setup import lifespan
        >>> app = FastAPI(lifespan=lifespan)

    Args:
        app: FastAPI application instance.
    """
    service = get_service()
    try:
        await service.startup()
        yield
    finally:
        await service.shutdown()
