# -*- coding: utf-8 -*-
"""
DDS Creator Service Facade - AGENT-EUDR-037

High-level service facade that wires together all 7 processing engines
into a single cohesive entry point. Provides the primary API used by
the FastAPI router layer to manage DDS creation, geolocation formatting,
risk data integration, supply chain compilation, compliance validation,
document packaging, version control, digital signing, amendment handling,
and EU Information System submission per EUDR Article 4.

Engines (7):
    1. StatementAssembler     - DDS assembly and lifecycle management
    2. GeolocationFormatter   - Article 9 geolocation data formatting
    3. RiskDataIntegrator     - Upstream risk assessment aggregation
    4. SupplyChainCompiler    - Supply chain traceability compilation
    5. ComplianceValidator    - Article 4 mandatory field validation
    6. DocumentPackager       - Evidence document packaging
    7. VersionController      - Version control, amendments, signatures

Singleton Pattern:
    Thread-safe singleton with double-checked locking via ``get_service()``.

FastAPI Integration:
    Use the ``lifespan`` async context manager with
    ``FastAPI(lifespan=lifespan)`` for automatic startup/shutdown.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-037 Due Diligence Statement Creator (GL-EUDR-DDSC-037)
Regulation: EU 2023/1115 (EUDR) Articles 4, 8, 9, 10, 12, 13, 14, 31
Status: Production Ready
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_VERSION = "1.0.0"
_AGENT_ID = "GL-EUDR-DDSC-037"

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
# Internal imports
# ---------------------------------------------------------------------------

from greenlang.agents.eudr.due_diligence_statement_creator.config import (
    DDSCreatorConfig,
    get_config,
)

try:
    from greenlang.agents.eudr.due_diligence_statement_creator.provenance import (
        ProvenanceTracker,
        GENESIS_HASH,
    )
except ImportError:
    ProvenanceTracker = None  # type: ignore[misc,assignment]
    GENESIS_HASH = "0" * 64

# Engine imports (conditional)
try:
    from greenlang.agents.eudr.due_diligence_statement_creator.statement_assembler import StatementAssembler
except ImportError:
    StatementAssembler = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.eudr.due_diligence_statement_creator.geolocation_formatter import GeolocationFormatter
except ImportError:
    GeolocationFormatter = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.eudr.due_diligence_statement_creator.risk_data_integrator import RiskDataIntegrator
except ImportError:
    RiskDataIntegrator = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.eudr.due_diligence_statement_creator.supply_chain_compiler import SupplyChainCompiler
except ImportError:
    SupplyChainCompiler = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.eudr.due_diligence_statement_creator.compliance_validator import ComplianceValidator
except ImportError:
    ComplianceValidator = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.eudr.due_diligence_statement_creator.document_packager import DocumentPackager
except ImportError:
    DocumentPackager = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.eudr.due_diligence_statement_creator.version_controller import VersionController
except ImportError:
    VersionController = None  # type: ignore[misc,assignment]

# Metrics imports (conditional)
try:
    from greenlang.agents.eudr.due_diligence_statement_creator.metrics import (
        record_statement_created,
        record_statement_submitted,
        record_amendment_created,
        record_validation_passed,
        record_validation_failed,
        record_document_packaged,
        record_signature_applied,
        record_geolocation_formatted,
        record_risk_integration,
        record_supply_chain_compilation,
        record_version_created,
        record_withdrawal,
        observe_statement_generation_duration,
        observe_validation_duration,
        observe_geolocation_formatting_duration,
        observe_risk_integration_duration,
        observe_supply_chain_compilation_duration,
        observe_document_packaging_duration,
        observe_signing_duration,
        observe_amendment_duration,
        observe_version_creation_duration,
        set_active_statements,
        set_pending_submissions,
        set_failed_validations,
        set_draft_statements,
    )
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False


def _safe_metric(fn: Any, *args: Any) -> None:
    """Safely call a metrics function if available."""
    if METRICS_AVAILABLE and fn is not None:
        try:
            fn(*args)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Service Facade
# ---------------------------------------------------------------------------


class DDSCreatorService:
    """Unified service facade for AGENT-EUDR-037.

    Aggregates all 7 processing engines and provides a clean API for
    Due Diligence Statement lifecycle management.

    Attributes:
        config: Agent configuration.
        _initialized: Whether startup has completed.
    """

    def __init__(
        self,
        config: Optional[DDSCreatorConfig] = None,
    ) -> None:
        """Initialize the service facade.

        Args:
            config: Optional configuration override.
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

        # Engine references
        self._assembler: Optional[Any] = None
        self._geolocation_formatter: Optional[Any] = None
        self._risk_integrator: Optional[Any] = None
        self._supply_chain_compiler: Optional[Any] = None
        self._compliance_validator: Optional[Any] = None
        self._document_packager: Optional[Any] = None
        self._version_controller: Optional[Any] = None
        self._engines: Dict[str, Any] = {}

        self._initialized = False
        logger.info("DDSCreatorService created")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        """Initialize all engines and external connections."""
        start = time.monotonic()
        logger.info("DDSCreatorService startup initiated")

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
                logger.warning("PostgreSQL pool init failed: %s", e)
                self._db_pool = None

        # Initialize Redis
        if REDIS_AVAILABLE:
            try:
                redis_url = (
                    f"redis://{self.config.redis_host}:"
                    f"{self.config.redis_port}/{self.config.redis_db}"
                )
                self._redis = aioredis.from_url(
                    redis_url, decode_responses=True,
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
            f"DDSCreatorService startup complete: "
            f"{engine_count}/7 engines in {elapsed:.1f}ms"
        )

    def _init_engines(self) -> None:
        """Initialize all 7 processing engines with graceful degradation."""
        engine_specs: List[Tuple[str, Any]] = [
            ("statement_assembler", StatementAssembler),
            ("geolocation_formatter", GeolocationFormatter),
            ("risk_data_integrator", RiskDataIntegrator),
            ("supply_chain_compiler", SupplyChainCompiler),
            ("compliance_validator", ComplianceValidator),
            ("document_packager", DocumentPackager),
            ("version_controller", VersionController),
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

        self._assembler = self._engines.get("statement_assembler")
        self._geolocation_formatter = self._engines.get("geolocation_formatter")
        self._risk_integrator = self._engines.get("risk_data_integrator")
        self._supply_chain_compiler = self._engines.get("supply_chain_compiler")
        self._compliance_validator = self._engines.get("compliance_validator")
        self._document_packager = self._engines.get("document_packager")
        self._version_controller = self._engines.get("version_controller")

    async def shutdown(self) -> None:
        """Gracefully shutdown all connections and engines."""
        logger.info("DDSCreatorService shutdown initiated")

        for name, engine in self._engines.items():
            if hasattr(engine, "shutdown"):
                try:
                    result = engine.shutdown()
                    if asyncio.iscoroutine(result):
                        await result
                    logger.info("Engine '%s' shut down", name)
                except Exception as e:
                    logger.warning("Engine '%s' shutdown error: %s", name, e)

        if self._redis is not None:
            try:
                await self._redis.close()
            except Exception as e:
                logger.warning("Redis close error: %s", e)

        if self._db_pool is not None:
            try:
                await self._db_pool.close()
            except Exception as e:
                logger.warning("PostgreSQL pool close error: %s", e)

        self._initialized = False
        logger.info("DDSCreatorService shutdown complete")

    # ------------------------------------------------------------------
    # Statement Assembly (Engine 1)
    # ------------------------------------------------------------------

    async def create_statement(
        self,
        operator_id: str,
        operator_name: str,
        commodities: List[str],
        statement_type: str = "placing",
        operator_address: str = "",
        operator_eori_number: str = "",
        language: str = "en",
    ) -> Any:
        """Create a new DDS in draft status."""
        if self._assembler is None:
            raise RuntimeError("StatementAssembler engine not available")
        result = await self._assembler.assemble_statement(
            operator_id=operator_id,
            operator_name=operator_name,
            commodities=commodities,
            statement_type=statement_type,
            operator_address=operator_address,
            operator_eori_number=operator_eori_number,
            language=language,
        )
        _safe_metric(record_statement_created, statement_type)
        return result

    async def assemble_statement(self, **kwargs: Any) -> Any:
        """Assemble a complete DDS with all provided data."""
        if self._assembler is None:
            raise RuntimeError("StatementAssembler engine not available")
        result = await self._assembler.assemble_statement(**kwargs)
        _safe_metric(
            record_statement_created,
            kwargs.get("statement_type", "placing"),
        )
        return result

    async def get_statement(self, statement_id: str) -> Any:
        """Get a DDS by its identifier."""
        if self._assembler is not None:
            return await self._assembler.get_statement(statement_id)
        return None

    async def list_statements(
        self,
        operator_id: Optional[str] = None,
        status: Optional[str] = None,
        commodity: Optional[str] = None,
    ) -> List[Any]:
        """List DDS records with optional filters."""
        if self._assembler is not None:
            return await self._assembler.list_statements(
                operator_id=operator_id, status=status, commodity=commodity,
            )
        return []

    async def get_statement_summary(self, statement_id: str) -> Any:
        """Get a lightweight summary of a DDS."""
        if self._assembler is not None:
            return await self._assembler.get_summary(statement_id)
        return None

    async def update_statement_status(
        self, statement_id: str, status: str,
    ) -> Any:
        """Update the status of a DDS."""
        if self._assembler is None:
            raise RuntimeError("StatementAssembler engine not available")
        return await self._assembler.update_status(statement_id, status)

    async def withdraw_statement(
        self, statement_id: str, reason: str = "",
    ) -> Any:
        """Withdraw a DDS."""
        if self._assembler is None:
            raise RuntimeError("StatementAssembler engine not available")
        result = await self._assembler.update_status(
            statement_id, "withdrawn",
        )
        _safe_metric(record_withdrawal)
        return result

    # ------------------------------------------------------------------
    # Geolocation Formatting (Engine 2)
    # ------------------------------------------------------------------

    async def format_geolocation(
        self,
        statement_id: str,
        plot_id: str,
        latitude: float,
        longitude: float,
        area_hectares: float = 0.0,
        polygon_coordinates: Optional[List[List[float]]] = None,
        country_code: str = "",
        collection_method: str = "gps_field_survey",
    ) -> Any:
        """Format geolocation data per Article 9."""
        if self._geolocation_formatter is None:
            raise RuntimeError("GeolocationFormatter engine not available")
        result = await self._geolocation_formatter.format_geolocation(
            plot_id=plot_id,
            latitude=latitude,
            longitude=longitude,
            area_hectares=area_hectares,
            polygon_coordinates=polygon_coordinates,
            country_code=country_code,
            collection_method=collection_method,
        )
        _safe_metric(record_geolocation_formatted, collection_method)
        return result

    async def format_geolocations_batch(
        self,
        statement_id: str,
        geolocations: List[Dict[str, Any]],
    ) -> List[Any]:
        """Format multiple geolocations in batch."""
        if self._geolocation_formatter is None:
            raise RuntimeError("GeolocationFormatter engine not available")
        return await self._geolocation_formatter.format_batch(geolocations)

    async def export_geojson(self, statement_id: str) -> Dict[str, Any]:
        """Export geolocation data as GeoJSON."""
        if self._geolocation_formatter is None:
            raise RuntimeError("GeolocationFormatter engine not available")
        stmt = await self.get_statement(statement_id)
        if stmt is None:
            raise ValueError(f"DDS {statement_id} not found")
        geos = stmt.geolocations if hasattr(stmt, "geolocations") else []
        return await self._geolocation_formatter.generate_geojson(geos)

    # ------------------------------------------------------------------
    # Risk Data Integration (Engine 3)
    # ------------------------------------------------------------------

    async def integrate_risk(
        self,
        statement_id: str,
        risk_id: str,
        source_agent: str,
        risk_category: str,
        risk_level: str = "standard",
        risk_score: float = 0.0,
        factors: Optional[List[str]] = None,
        mitigation_measures: Optional[List[str]] = None,
        data_sources: Optional[List[str]] = None,
    ) -> Any:
        """Integrate a risk assessment reference."""
        if self._risk_integrator is None:
            raise RuntimeError("RiskDataIntegrator engine not available")
        result = await self._risk_integrator.integrate_risk(
            risk_id=risk_id,
            source_agent=source_agent,
            risk_category=risk_category,
            risk_level=risk_level,
            risk_score=risk_score,
            factors=factors,
            mitigation_measures=mitigation_measures,
            data_sources=data_sources,
        )
        _safe_metric(record_risk_integration, source_agent)
        return result

    async def integrate_risk_batch(
        self,
        statement_id: str,
        risk_data: List[Dict[str, Any]],
    ) -> List[Any]:
        """Integrate multiple risk assessments in batch."""
        if self._risk_integrator is None:
            raise RuntimeError("RiskDataIntegrator engine not available")
        return await self._risk_integrator.integrate_batch(risk_data)

    async def get_overall_risk(
        self, statement_id: str,
    ) -> Dict[str, Any]:
        """Get the overall risk level for a DDS."""
        if self._risk_integrator is None:
            raise RuntimeError("RiskDataIntegrator engine not available")
        stmt = await self.get_statement(statement_id)
        if stmt is None:
            raise ValueError(f"DDS {statement_id} not found")
        refs = stmt.risk_references if hasattr(stmt, "risk_references") else []
        level = await self._risk_integrator.compute_overall_risk(refs)
        score = await self._risk_integrator.aggregate_scores(refs)
        return {
            "statement_id": statement_id,
            "overall_risk_level": level.value,
            "average_risk_score": float(score),
            "reference_count": len(refs),
        }

    # ------------------------------------------------------------------
    # Supply Chain Compilation (Engine 4)
    # ------------------------------------------------------------------

    async def compile_supply_chain(
        self,
        statement_id: str,
        supply_chain_id: str,
        commodity: str,
        suppliers: Optional[List[Dict[str, Any]]] = None,
        countries_of_production: Optional[List[str]] = None,
        chain_of_custody_model: str = "segregation",
        traceability_score: float = 0.0,
    ) -> Any:
        """Compile supply chain data."""
        if self._supply_chain_compiler is None:
            raise RuntimeError("SupplyChainCompiler engine not available")
        stmt = await self.get_statement(statement_id)
        operator_id = stmt.operator_id if stmt else ""
        result = await self._supply_chain_compiler.compile_supply_chain(
            supply_chain_id=supply_chain_id,
            operator_id=operator_id,
            commodity=commodity,
            suppliers=suppliers,
            countries_of_production=countries_of_production,
            chain_of_custody_model=chain_of_custody_model,
            traceability_score=traceability_score,
        )
        _safe_metric(record_supply_chain_compilation)
        return result

    async def validate_supply_chain(
        self, statement_id: str,
    ) -> Dict[str, Any]:
        """Validate supply chain completeness."""
        if self._supply_chain_compiler is None:
            raise RuntimeError("SupplyChainCompiler engine not available")
        stmt = await self.get_statement(statement_id)
        if stmt is None:
            raise ValueError(f"DDS {statement_id} not found")
        sc = stmt.supply_chain_data if hasattr(stmt, "supply_chain_data") else None
        if sc is None:
            return {"complete": False, "issues": ["No supply chain data"]}
        return await self._supply_chain_compiler.validate_completeness(sc)

    async def get_supply_chain_countries(
        self, statement_id: str,
    ) -> Dict[str, int]:
        """Get supplier counts by country."""
        if self._supply_chain_compiler is None:
            raise RuntimeError("SupplyChainCompiler engine not available")
        stmt = await self.get_statement(statement_id)
        if stmt is None:
            raise ValueError(f"DDS {statement_id} not found")
        sc = stmt.supply_chain_data if hasattr(stmt, "supply_chain_data") else None
        if sc is None:
            return {}
        return await self._supply_chain_compiler.get_countries_summary(sc)

    # ------------------------------------------------------------------
    # Compliance Validation (Engine 5)
    # ------------------------------------------------------------------

    async def validate_statement(self, statement_id: str) -> Any:
        """Validate a DDS against Article 4 requirements."""
        if self._compliance_validator is None:
            raise RuntimeError("ComplianceValidator engine not available")
        stmt = await self.get_statement(statement_id)
        if stmt is None:
            raise ValueError(f"DDS {statement_id} not found")
        result = await self._compliance_validator.validate_statement(stmt)
        if hasattr(result, "overall_result"):
            if result.overall_result.value == "pass":
                _safe_metric(record_validation_passed)
            else:
                _safe_metric(record_validation_failed, "article_4")
        return result

    async def get_compliance_report(self, statement_id: str) -> Any:
        """Get the latest compliance validation report."""
        return await self.validate_statement(statement_id)

    # ------------------------------------------------------------------
    # Document Packaging (Engine 6)
    # ------------------------------------------------------------------

    async def add_document(
        self,
        statement_id: str,
        document_type: str,
        filename: str,
        size_bytes: int = 0,
        mime_type: str = "application/pdf",
        hash_sha256: str = "",
        description: str = "",
        issuing_authority: str = "",
        language: str = "en",
    ) -> Any:
        """Add a supporting document."""
        if self._document_packager is None:
            raise RuntimeError("DocumentPackager engine not available")
        result = await self._document_packager.add_document(
            document_type=document_type,
            filename=filename,
            size_bytes=size_bytes,
            mime_type=mime_type,
            hash_sha256=hash_sha256,
            description=description,
            issuing_authority=issuing_authority,
            language=language,
        )
        _safe_metric(record_document_packaged, document_type)
        return result

    async def create_submission_package(self, statement_id: str) -> Any:
        """Create a submission package for the EU IS."""
        if self._document_packager is None:
            raise RuntimeError("DocumentPackager engine not available")
        stmt = await self.get_statement(statement_id)
        if stmt is None:
            raise ValueError(f"DDS {statement_id} not found")
        return await self._document_packager.create_submission_package(stmt)

    async def validate_package(self, statement_id: str) -> Dict[str, Any]:
        """Validate a submission package."""
        if self._document_packager is None:
            raise RuntimeError("DocumentPackager engine not available")
        stmt = await self.get_statement(statement_id)
        if stmt is None:
            raise ValueError(f"DDS {statement_id} not found")
        pkg = stmt.submission_package if hasattr(stmt, "submission_package") else None
        if pkg is None:
            return {"valid": False, "issues": ["No submission package created"]}
        return await self._document_packager.validate_package(pkg)

    async def get_package_manifest(
        self, statement_id: str,
    ) -> Dict[str, Any]:
        """Get the document manifest for a package."""
        if self._document_packager is None:
            raise RuntimeError("DocumentPackager engine not available")
        stmt = await self.get_statement(statement_id)
        if stmt is None:
            raise ValueError(f"DDS {statement_id} not found")
        docs = stmt.supporting_documents if hasattr(stmt, "supporting_documents") else []
        return await self._document_packager.get_manifest(docs)

    # ------------------------------------------------------------------
    # Version Control & Signatures (Engine 7)
    # ------------------------------------------------------------------

    async def apply_signature(
        self,
        statement_id: str,
        signer_name: str,
        signer_role: str = "",
        signer_organization: str = "",
        signature_type: str = "qualified_electronic",
        signed_hash: str = "",
    ) -> Any:
        """Apply a digital signature to a DDS."""
        if self._version_controller is None:
            raise RuntimeError("VersionController engine not available")
        result = await self._version_controller.apply_signature(
            statement_id=statement_id,
            signer_name=signer_name,
            signer_role=signer_role,
            signer_organization=signer_organization,
            signature_type=signature_type,
            signed_hash=signed_hash,
        )
        _safe_metric(record_signature_applied, signature_type)
        return result

    async def validate_signature(
        self, statement_id: str,
    ) -> Dict[str, Any]:
        """Validate the digital signature on a DDS."""
        if self._version_controller is None:
            raise RuntimeError("VersionController engine not available")
        stmt = await self.get_statement(statement_id)
        if stmt is None:
            raise ValueError(f"DDS {statement_id} not found")
        sig = stmt.digital_signature if hasattr(stmt, "digital_signature") else None
        if sig is None:
            return {"valid": False, "issues": ["No digital signature applied"]}
        return await self._version_controller.validate_signature(sig)

    async def create_amendment(
        self,
        statement_id: str,
        reason: str,
        description: str,
        previous_version: int,
        changed_fields: Optional[List[str]] = None,
        changed_by: str = "",
        approved_by: str = "",
    ) -> Any:
        """Create an amendment to a DDS."""
        if self._version_controller is None:
            raise RuntimeError("VersionController engine not available")
        result = await self._version_controller.create_amendment(
            statement_id=statement_id,
            reason=reason,
            description=description,
            previous_version=previous_version,
            changed_fields=changed_fields,
            changed_by=changed_by,
            approved_by=approved_by,
        )
        _safe_metric(record_amendment_created, reason)
        _safe_metric(record_version_created)
        return result

    async def get_versions(self, statement_id: str) -> List[Any]:
        """Get version history for a DDS."""
        if self._version_controller is not None:
            return await self._version_controller.get_versions(statement_id)
        return []

    async def get_latest_version(self, statement_id: str) -> Any:
        """Get the latest version of a DDS."""
        if self._version_controller is not None:
            return await self._version_controller.get_latest_version(
                statement_id,
            )
        return None

    async def get_amendments(self, statement_id: str) -> List[Any]:
        """Get amendment records for a DDS."""
        if self._version_controller is not None:
            return await self._version_controller.get_amendments(statement_id)
        return []

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    async def submit_statement(
        self,
        statement_id: str,
        additional_documents: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:
        """Submit a DDS to the EU Information System.

        Orchestrates validation, packaging, and status update.
        """
        # Validate first
        validation = await self.validate_statement(statement_id)
        if hasattr(validation, "overall_result"):
            if validation.overall_result.value == "fail":
                raise ValueError(
                    f"DDS {statement_id} failed compliance validation"
                )

        # Create package
        pkg = await self.create_submission_package(statement_id)

        # Update status
        await self.update_statement_status(statement_id, "submitted")
        _safe_metric(record_statement_submitted, "submitted")

        return {
            "statement_id": statement_id,
            "status": "submitted",
            "package": (
                pkg.model_dump(mode="json")
                if hasattr(pkg, "model_dump") else pkg
            ),
        }

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def health_check(self) -> Dict[str, Any]:
        """Perform a comprehensive health check."""
        result: Dict[str, Any] = {
            "agent_id": _AGENT_ID,
            "version": _VERSION,
            "status": "healthy",
            "initialized": self._initialized,
            "engines": {},
            "connections": {},
            "timestamp": None,
        }

        from datetime import datetime, timezone
        result["timestamp"] = datetime.now(timezone.utc).replace(
            microsecond=0,
        ).isoformat()

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
            "statement_assembler",
            "geolocation_formatter",
            "risk_data_integrator",
            "supply_chain_compiler",
            "compliance_validator",
            "document_packager",
            "version_controller",
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
            1 for v in result["engines"].values()
            if isinstance(v, dict) and v.get("status") in (
                "error", "not_loaded",
            )
        )
        if unhealthy_engines > 3:
            result["status"] = "unhealthy"
        elif unhealthy_engines > 0:
            result["status"] = "degraded"

        return result

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
_service_instance: Optional[DDSCreatorService] = None


def get_service(
    config: Optional[DDSCreatorConfig] = None,
) -> DDSCreatorService:
    """Get the global DDSCreatorService singleton instance.

    Thread-safe lazy initialization.

    Args:
        config: Optional configuration override for first creation.

    Returns:
        DDSCreatorService singleton instance.
    """
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = DDSCreatorService(config)
    return _service_instance


def reset_service() -> None:
    """Reset the global service singleton to None (testing only)."""
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
        >>> from greenlang.agents.eudr.due_diligence_statement_creator.setup import lifespan
        >>> app = FastAPI(lifespan=lifespan)

    Args:
        app: FastAPI application instance.

    Yields:
        None -- application runs between startup and shutdown.
    """
    service = get_service()
    await service.startup()
    logger.info("DDS Creator lifespan: startup complete")

    try:
        yield
    finally:
        await service.shutdown()
        logger.info("DDS Creator lifespan: shutdown complete")
