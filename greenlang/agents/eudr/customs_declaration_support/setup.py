# -*- coding: utf-8 -*-
"""
Customs Declaration Support Service Facade - AGENT-EUDR-039

High-level service facade that wires together all 7 processing engines
into a single cohesive entry point. Provides the primary API used by
the FastAPI router layer to manage customs declarations, CN/HS code
operations, origin verification, value calculation, compliance checking,
and customs system submission per EUDR and UCC requirements.

Engines (7):
    1. CNCodeMapper              - EUDR commodity to CN code mapping
    2. HSCodeValidator           - HS code validation engine
    3. DeclarationGenerator      - SAD form and declaration generation
    4. OriginValidator           - Country of origin verification
    5. ValueCalculator           - CIF/FOB customs value calculation
    6. ComplianceChecker         - EUDR Article 4 compliance verification
    7. CustomsInterface          - NCTS/AIS/ICS2 submission interface

Singleton Pattern:
    Thread-safe singleton with double-checked locking via ``get_service()``.

FastAPI Integration:
    Use the ``lifespan`` async context manager with
    ``FastAPI(lifespan=lifespan)`` for automatic startup/shutdown.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-039 Customs Declaration Support (GL-EUDR-CDS-039)
Regulation: EU 2023/1115 (EUDR) Articles 4, 5, 6, 12, 31; EU UCC 952/2013
Status: Production Ready
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from contextlib import asynccontextmanager
from decimal import Decimal
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_VERSION = "1.0.0"
_AGENT_ID = "GL-EUDR-CDS-039"

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

from greenlang.agents.eudr.customs_declaration_support.config import (
    CustomsDeclarationSupportConfig,
    get_config,
)

try:
    from greenlang.agents.eudr.customs_declaration_support.provenance import (
        ProvenanceTracker,
        GENESIS_HASH,
    )
except ImportError:
    ProvenanceTracker = None  # type: ignore[misc,assignment]
    GENESIS_HASH = "0000000000000000000000000000000000000000000000000000000000000000"

# Engine imports (conditional)
try:
    from greenlang.agents.eudr.customs_declaration_support.cn_code_mapper import CNCodeMapper
except ImportError:
    CNCodeMapper = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.eudr.customs_declaration_support.hs_code_validator import HSCodeValidator
except ImportError:
    HSCodeValidator = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.eudr.customs_declaration_support.declaration_generator import DeclarationGenerator
except ImportError:
    DeclarationGenerator = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.eudr.customs_declaration_support.origin_validator import OriginValidator
except ImportError:
    OriginValidator = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.eudr.customs_declaration_support.value_calculator import ValueCalculator
except ImportError:
    ValueCalculator = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.eudr.customs_declaration_support.compliance_checker import ComplianceChecker
except ImportError:
    ComplianceChecker = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.eudr.customs_declaration_support.customs_interface import CustomsInterface
except ImportError:
    CustomsInterface = None  # type: ignore[misc,assignment]

# Metrics imports (conditional)
try:
    from greenlang.agents.eudr.customs_declaration_support.metrics import (
        record_declaration_created,
        record_declaration_submitted,
        record_declaration_cleared,
        record_compliance_check_passed,
        record_compliance_check_failed,
        record_tariff_calculated,
        record_cn_code_mapped,
        record_hs_code_validated,
        record_origin_verification,
        record_value_calculation,
        record_mrn_assigned,
        record_sad_form_generated,
        observe_declaration_generation_duration,
        observe_submission_duration,
        observe_compliance_check_duration,
        observe_tariff_calculation_duration,
        observe_value_calculation_duration,
        observe_cn_code_mapping_duration,
        observe_hs_code_validation_duration,
        observe_origin_verification_duration,
        set_pending_declarations,
        set_declarations_awaiting_clearance,
        set_total_customs_value_eur,
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


class CustomsDeclarationService:
    """Unified service facade for AGENT-EUDR-039.

    Aggregates all 7 processing engines and provides a clean API for
    customs declaration lifecycle management.

    Attributes:
        config: Agent configuration.
        _initialized: Whether startup has completed.
    """

    def __init__(
        self,
        config: Optional[CustomsDeclarationSupportConfig] = None,
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
        self._cn_code_mapper: Optional[Any] = None
        self._hs_code_validator: Optional[Any] = None
        self._declaration_generator: Optional[Any] = None
        self._origin_validator: Optional[Any] = None
        self._value_calculator: Optional[Any] = None
        self._compliance_checker: Optional[Any] = None
        self._customs_interface: Optional[Any] = None
        self._engines: Dict[str, Any] = {}

        self._initialized = False
        logger.info("CustomsDeclarationService created")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        """Initialize all engines and external connections."""
        start = time.monotonic()
        logger.info("CustomsDeclarationService startup initiated")

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
            f"CustomsDeclarationService startup complete: "
            f"{engine_count}/7 engines in {elapsed:.1f}ms"
        )

    def _init_engines(self) -> None:
        """Initialize all 7 processing engines with graceful degradation."""
        engine_specs: List[Tuple[str, Any]] = [
            ("cn_code_mapper", CNCodeMapper),
            ("hs_code_validator", HSCodeValidator),
            ("declaration_generator", DeclarationGenerator),
            ("origin_validator", OriginValidator),
            ("value_calculator", ValueCalculator),
            ("compliance_checker", ComplianceChecker),
            ("customs_interface", CustomsInterface),
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

        self._cn_code_mapper = self._engines.get("cn_code_mapper")
        self._hs_code_validator = self._engines.get("hs_code_validator")
        self._declaration_generator = self._engines.get("declaration_generator")
        self._origin_validator = self._engines.get("origin_validator")
        self._value_calculator = self._engines.get("value_calculator")
        self._compliance_checker = self._engines.get("compliance_checker")
        self._customs_interface = self._engines.get("customs_interface")

    async def shutdown(self) -> None:
        """Gracefully shutdown all connections and engines."""
        logger.info("CustomsDeclarationService shutdown initiated")

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
        logger.info("CustomsDeclarationService shutdown complete")

    # ------------------------------------------------------------------
    # Declaration Operations
    # ------------------------------------------------------------------

    async def create_declaration(
        self,
        operator_id: str = "",
        declaration_data: Optional[Dict[str, Any]] = None,
        *,
        commodities: Optional[List[str]] = None,
        country_of_origin: str = "",
        **kwargs: Any,
    ) -> Any:
        """Create a new customs declaration."""
        if self._declaration_generator is None:
            raise RuntimeError("DeclarationGenerator engine not available")
        result = await self._declaration_generator.create_declaration(
            operator_id=operator_id,
            declaration_data=declaration_data,
            commodities=commodities,
            country_of_origin=country_of_origin,
            **kwargs,
        )
        _safe_metric(
            record_declaration_created,
            (declaration_data or {}).get("declaration_type", "import"),
        )
        return result

    async def add_line_item(
        self, declaration_id: str, line_data: Dict[str, Any],
    ) -> Any:
        """Add a line item to a declaration."""
        if self._declaration_generator is not None:
            return await self._declaration_generator.add_line_item(
                declaration_id=declaration_id, line_data=line_data,
            )
        raise RuntimeError("DeclarationGenerator engine not available")

    async def validate_declaration(self, declaration_id: str) -> Dict[str, Any]:
        """Validate a declaration."""
        if self._declaration_generator is not None:
            return await self._declaration_generator.validate_declaration(
                declaration_id,
            )
        raise RuntimeError("DeclarationGenerator engine not available")

    async def submit_declaration(
        self,
        declaration_id: str = "",
        customs_system: Optional[str] = None,
        *,
        system: Optional[str] = None,
    ) -> Any:
        """Submit a declaration to customs authority."""
        if self._customs_interface is None:
            raise RuntimeError("CustomsInterface engine not available")

        # Determine target system
        target_system = system or customs_system

        # Try to get declaration from generator if available
        declaration = None
        if self._declaration_generator is not None:
            declaration = await self._declaration_generator.get_declaration(
                declaration_id,
            )

        if declaration is not None:
            result = await self._customs_interface.submit_declaration(
                declaration=declaration, customs_system=target_system,
            )
            _safe_metric(
                record_declaration_submitted,
                target_system or declaration.customs_system.value,
            )
            if result.mrn:
                _safe_metric(record_mrn_assigned)
            return result

        raise ValueError(f"Declaration '{declaration_id}' not found")

    async def get_declaration(self, declaration_id: str) -> Any:
        """Get declaration by ID."""
        if self._declaration_generator is not None:
            return await self._declaration_generator.get_declaration(
                declaration_id,
            )
        return None

    async def list_declarations(
        self, operator_id: Optional[str] = None,
        status: Optional[str] = None,
        declaration_type: Optional[str] = None,
    ) -> List[Any]:
        """List declarations with filters."""
        if self._declaration_generator is not None:
            return await self._declaration_generator.list_declarations(
                operator_id=operator_id, status=status,
                declaration_type=declaration_type,
            )
        return []

    async def check_clearance_status(
        self, declaration_id: str, mrn: str = "",
    ) -> Dict[str, Any]:
        """Check clearance status."""
        if self._customs_interface is not None:
            return await self._customs_interface.check_clearance_status(
                declaration_id, mrn,
            )
        return {"declaration_id": declaration_id, "status": "unknown"}

    async def amend_declaration(self, declaration_id: str) -> Dict[str, Any]:
        """Amend a submitted declaration."""
        if self._declaration_generator is not None:
            decl = await self._declaration_generator.get_declaration(
                declaration_id,
            )
            if decl is not None:
                from .models import DeclarationStatus
                decl.status = DeclarationStatus.AMENDED
                return {"declaration_id": declaration_id, "status": "amended"}
        raise ValueError(f"Declaration '{declaration_id}' not found")

    async def cancel_declaration(
        self, declaration_id: str, reason: str = "",
    ) -> Dict[str, Any]:
        """Cancel a declaration."""
        if self._declaration_generator is not None:
            decl = await self._declaration_generator.get_declaration(
                declaration_id,
            )
            if decl is not None:
                from .models import DeclarationStatus
                decl.status = DeclarationStatus.CANCELLED
                return {
                    "declaration_id": declaration_id,
                    "status": "cancelled",
                    "reason": reason,
                }
        raise ValueError(f"Declaration '{declaration_id}' not found")

    async def update_declaration_status(
        self,
        declaration_id: str,
        status: str,
    ) -> Dict[str, Any]:
        """Update a declaration's status."""
        if self._declaration_generator is not None:
            return await self._declaration_generator.update_status(
                declaration_id, status,
            )
        raise RuntimeError("DeclarationGenerator engine not available")

    async def map_cn_codes(
        self,
        commodity: str = "",
        **kwargs: Any,
    ) -> Any:
        """Map commodity to CN codes (convenience alias)."""
        if self._cn_code_mapper is not None:
            return await self._cn_code_mapper.map_commodity(commodity)
        raise RuntimeError("CNCodeMapper engine not available")

    async def lookup_cn_code(self, cn_code: str) -> Any:
        """Lookup a CN code."""
        if self._cn_code_mapper is not None:
            return await self._cn_code_mapper.lookup_cn_code(cn_code)
        return None

    async def validate_hs_codes_batch(self, hs_codes: List[str]) -> List[Any]:
        """Batch validate HS codes."""
        if self._hs_code_validator is not None:
            return await self._hs_code_validator.batch_validate(hs_codes)
        return []

    async def calculate_tariff(
        self,
        declaration_id: str = "",
        cn_code: str = "",
        customs_value: float = 0.0,
        quantity: float = 0.0,
        origin_country: str = "",
        **kwargs: Any,
    ) -> Any:
        """Calculate tariff for a declaration."""
        if self._value_calculator is not None:
            return await self._value_calculator.calculate_tariff(
                cn_code=cn_code,
                customs_value=Decimal(str(customs_value)),
                origin_country=origin_country,
            )
        raise RuntimeError("ValueCalculator engine not available")

    async def run_compliance_check(
        self,
        declaration_id: str = "",
        dds_reference: str = "",
        cn_codes: Optional[List[str]] = None,
        declared_origin: str = "",
        **kwargs: Any,
    ) -> Any:
        """Run compliance checks (convenience alias)."""
        if self._compliance_checker is not None:
            return await self._compliance_checker.run_full_compliance_check(
                declaration_id=declaration_id,
                dds_reference=dds_reference,
                cn_codes=cn_codes,
                declared_origin=declared_origin,
                **kwargs,
            )
        raise RuntimeError("ComplianceChecker engine not available")

    async def check_dds_reference(
        self,
        declaration_id: str = "",
        dds_reference: str = "",
    ) -> Any:
        """Check DDS reference (convenience alias)."""
        if self._compliance_checker is not None:
            return await self._compliance_checker.check_dds_reference(
                declaration_id=declaration_id,
                dds_reference=dds_reference,
            )
        raise RuntimeError("ComplianceChecker engine not available")

    async def generate_sad_form(
        self,
        declaration_id: str = "",
        sad_data: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Generate a SAD form for a declaration."""
        if self._declaration_generator is not None:
            result = await self._declaration_generator.generate_sad_form(
                declaration_id=declaration_id, sad_data=sad_data,
            )
            _safe_metric(record_sad_form_generated, "IM")
            return result
        raise RuntimeError("DeclarationGenerator engine not available")

    async def submit_to_ncts(self, **kwargs: Any) -> Any:
        """Submit to NCTS (convenience alias)."""
        if self._customs_interface is not None:
            return await self._customs_interface.submit_to_ncts(**kwargs)
        raise RuntimeError("CustomsInterface engine not available")

    async def submit_to_ais(self, **kwargs: Any) -> Any:
        """Submit to AIS (convenience alias)."""
        if self._customs_interface is not None:
            return await self._customs_interface.submit_to_ais(**kwargs)
        raise RuntimeError("CustomsInterface engine not available")

    async def check_submission_status(self, **kwargs: Any) -> Any:
        """Check submission status (convenience alias)."""
        if self._customs_interface is not None:
            return await self._customs_interface.check_status(**kwargs)
        return None

    async def get_mrn_status(self, mrn: str = "") -> Any:
        """Get MRN status (convenience alias)."""
        if self._customs_interface is not None:
            return await self._customs_interface.check_status(mrn=mrn)
        return None

    async def get_tariff_summary(self, declaration_id: str = "") -> Any:
        """Get tariff summary for a declaration."""
        if self._value_calculator is not None:
            return {"declaration_id": declaration_id, "total_value": "0.00"}
        return None

    async def get_compliance_report(self, declaration_id: str = "") -> Any:
        """Get compliance report for a declaration."""
        if self._compliance_checker is not None:
            checks = self._compliance_checker._checks.get(declaration_id)
            if checks:
                return await self._compliance_checker.check_overall_compliance(declaration_id)
            return None
        return None

    async def verify_origin_batch(self, origins: List[Any] = None) -> List[Any]:
        """Batch verify origins."""
        if self._origin_validator is not None:
            return await self._origin_validator.verify_origin_batch(origins or [])
        return []

    # ------------------------------------------------------------------
    # CN Code Operations (Legacy interface)
    # ------------------------------------------------------------------

    async def map_commodity_to_cn_codes(
        self, commodity_type: str, product_description: str = "",
    ) -> List[Any]:
        """Map commodity to CN codes."""
        if self._cn_code_mapper is not None:
            result = await self._cn_code_mapper.map_commodity_to_cn_codes(
                commodity_type=commodity_type,
                product_description=product_description,
            )
            _safe_metric(record_cn_code_mapped, commodity_type)
            return result
        raise RuntimeError("CNCodeMapper engine not available")

    async def get_cn_code_details(self, cn_code: str) -> Any:
        """Get CN code details."""
        if self._cn_code_mapper is not None:
            return await self._cn_code_mapper.get_cn_code_details(cn_code)
        return None

    async def search_cn_codes(
        self, query: str, commodity_filter: Optional[str] = None,
        limit: int = 20,
    ) -> List[Any]:
        """Search CN codes."""
        if self._cn_code_mapper is not None:
            return await self._cn_code_mapper.search_cn_codes(
                query=query, commodity_filter=commodity_filter, limit=limit,
            )
        return []

    async def get_tariff_rate(self, cn_code: str) -> Decimal:
        """Get tariff rate for a CN code."""
        if self._cn_code_mapper is not None:
            return await self._cn_code_mapper.get_tariff_rate(cn_code)
        return Decimal("0")

    # ------------------------------------------------------------------
    # HS Code Operations
    # ------------------------------------------------------------------

    async def validate_hs_code(self, hs_code: str) -> Any:
        """Validate an HS code."""
        if self._hs_code_validator is not None:
            result = await self._hs_code_validator.validate_hs_code(hs_code)
            _safe_metric(record_hs_code_validated)
            return result
        raise RuntimeError("HSCodeValidator engine not available")

    async def batch_validate_hs_codes(self, hs_codes: List[str]) -> List[Any]:
        """Batch validate HS codes."""
        if self._hs_code_validator is not None:
            return await self._hs_code_validator.batch_validate(hs_codes)
        return []

    # ------------------------------------------------------------------
    # Origin Operations
    # ------------------------------------------------------------------

    async def verify_origin(
        self,
        declared_origin: str = "",
        supply_chain_data: Optional[Dict[str, Any]] = None,
        certificate_ref: str = "",
        *,
        declaration_id: str = "",
        supply_chain_origins: Optional[List[str]] = None,
        dds_reference: str = "",
    ) -> Any:
        """Verify country of origin."""
        if self._origin_validator is None:
            raise RuntimeError("OriginValidator engine not available")
        result = await self._origin_validator.verify_origin(
            declared_origin=declared_origin,
            supply_chain_data=supply_chain_data,
            certificate_ref=certificate_ref,
            declaration_id=declaration_id,
            supply_chain_origins=supply_chain_origins,
            dds_reference=dds_reference,
        )
        _safe_metric(record_origin_verification, result.verification_status.value)
        return result

    async def get_origin_verification(self, verification_id: str) -> Any:
        """Get origin verification result."""
        if self._origin_validator is not None:
            return await self._origin_validator.get_verification(
                verification_id,
            )
        return None

    async def batch_verify_origins(
        self, origins: List[Dict[str, Any]],
    ) -> List[Any]:
        """Batch verify origins."""
        if self._origin_validator is not None:
            return await self._origin_validator.batch_verify_origins(origins)
        return []

    # ------------------------------------------------------------------
    # Value Operations
    # ------------------------------------------------------------------

    async def calculate_customs_value(
        self, transaction_value: Decimal, currency: str = "EUR",
        incoterms: str = "CIF",
        freight_cost: Optional[Decimal] = None,
        insurance_cost: Optional[Decimal] = None,
        loading_cost: Optional[Decimal] = None,
        adjustments: Decimal = Decimal("0"),
    ) -> Any:
        """Calculate customs value."""
        if self._value_calculator is not None:
            result = await self._value_calculator.calculate_customs_value(
                transaction_value=transaction_value, currency=currency,
                incoterms=incoterms, freight_cost=freight_cost,
                insurance_cost=insurance_cost, loading_cost=loading_cost,
                adjustments=adjustments,
            )
            _safe_metric(record_value_calculation, incoterms)
            return result
        raise RuntimeError("ValueCalculator engine not available")

    async def convert_currency(
        self, amount: Decimal, from_currency: str,
        to_currency: str = "EUR",
    ) -> Decimal:
        """Convert currency."""
        if self._value_calculator is not None:
            return await self._value_calculator.convert_currency(
                amount=amount, from_currency=from_currency,
                to_currency=to_currency,
            )
        raise RuntimeError("ValueCalculator engine not available")

    async def get_value_calculation(self, value_id: str) -> Any:
        """Get value calculation."""
        if self._value_calculator is not None:
            return await self._value_calculator.get_calculation(value_id)
        return None

    async def get_exchange_rates(self) -> Dict[str, Decimal]:
        """Get all exchange rates."""
        if self._value_calculator is not None:
            return await self._value_calculator.get_all_exchange_rates()
        return {}

    # ------------------------------------------------------------------
    # Compliance Operations
    # ------------------------------------------------------------------

    async def run_compliance_checks(
        self, declaration_id: str, compliance_data: Dict[str, Any],
    ) -> List[Any]:
        """Run compliance checks."""
        if self._compliance_checker is not None:
            return await self._compliance_checker.run_compliance_checks(
                declaration_id=declaration_id,
                compliance_data=compliance_data,
            )
        raise RuntimeError("ComplianceChecker engine not available")

    async def get_compliance_status(
        self, declaration_id: str,
    ) -> Dict[str, Any]:
        """Get compliance status."""
        if self._compliance_checker is not None:
            return await self._compliance_checker.check_overall_compliance(
                declaration_id,
            )
        return {"declaration_id": declaration_id, "overall_status": "unknown"}

    # ------------------------------------------------------------------
    # Reference Data
    # ------------------------------------------------------------------

    async def list_ports(self) -> List[Dict[str, Any]]:
        """List EU ports of entry."""
        # Reference data - major EU ports handling EUDR commodities
        return [
            {"port_code": "NLRTM", "port_name": "Rotterdam", "country_code": "NL", "port_type": "sea"},
            {"port_code": "DEHAM", "port_name": "Hamburg", "country_code": "DE", "port_type": "sea"},
            {"port_code": "BEANR", "port_name": "Antwerp", "country_code": "BE", "port_type": "sea"},
            {"port_code": "FRLEH", "port_name": "Le Havre", "country_code": "FR", "port_type": "sea"},
            {"port_code": "ESBCN", "port_name": "Barcelona", "country_code": "ES", "port_type": "sea"},
            {"port_code": "ITGOA", "port_name": "Genoa", "country_code": "IT", "port_type": "sea"},
            {"port_code": "GRGLO", "port_name": "Piraeus", "country_code": "GR", "port_type": "sea"},
            {"port_code": "PLGDN", "port_name": "Gdansk", "country_code": "PL", "port_type": "sea"},
            {"port_code": "DEFRA", "port_name": "Frankfurt Airport", "country_code": "DE", "port_type": "air"},
            {"port_code": "NLAMS", "port_name": "Amsterdam Schiphol", "country_code": "NL", "port_type": "air"},
            {"port_code": "FRCDG", "port_name": "Paris CDG", "country_code": "FR", "port_type": "air"},
            {"port_code": "BEMPO", "port_name": "Brussels Zaventem", "country_code": "BE", "port_type": "air"},
        ]

    async def list_commodities(self) -> List[Dict[str, Any]]:
        """List EUDR-regulated commodities."""
        from .models import CommodityType, EUDR_COMMODITY_CN_CODES
        return [
            {
                "commodity_type": ct.value,
                "description": ct.value.replace("_", " ").title(),
                "cn_code_count": len(EUDR_COMMODITY_CN_CODES.get(ct.value, [])),
            }
            for ct in CommodityType
        ]

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
            microsecond=0
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
            "cn_code_mapper",
            "hs_code_validator",
            "declaration_generator",
            "origin_validator",
            "value_calculator",
            "compliance_checker",
            "customs_interface",
        ]

        for engine_name in expected_engines:
            if engine_name in self._engines:
                result["engines"][engine_name] = {"status": "available"}
            else:
                result["engines"][engine_name] = {"status": "not_loaded"}

        # Determine overall status
        unhealthy_engines = sum(
            1 for v in result["engines"].values()
            if isinstance(v, dict) and v.get("status") in ("error", "not_loaded")
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
_service_instance: Optional[CustomsDeclarationService] = None


def get_service(
    config: Optional[CustomsDeclarationSupportConfig] = None,
) -> CustomsDeclarationService:
    """Get the global CustomsDeclarationService singleton instance.

    Thread-safe lazy initialization.

    Args:
        config: Optional configuration override for first creation.

    Returns:
        CustomsDeclarationService singleton instance.
    """
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = CustomsDeclarationService(config)
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
        >>> from greenlang.agents.eudr.customs_declaration_support.setup import lifespan
        >>> app = FastAPI(lifespan=lifespan)

    Args:
        app: FastAPI application instance.

    Yields:
        None -- application runs between startup and shutdown.
    """
    service = get_service()
    await service.startup()
    logger.info("Customs Declaration Support lifespan: startup complete")

    try:
        yield
    finally:
        await service.shutdown()
        logger.info("Customs Declaration Support lifespan: shutdown complete")
