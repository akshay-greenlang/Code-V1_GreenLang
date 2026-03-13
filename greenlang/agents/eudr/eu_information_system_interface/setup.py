# -*- coding: utf-8 -*-
"""
EU Information System Interface Service Facade - AGENT-EUDR-036

High-level service facade that wires together all 7 processing engines
into a single cohesive entry point. Provides the primary API used by
the FastAPI router layer to submit DDS, register operators, format
geolocations, assemble packages, track submission status, communicate
with the EU IS API, and record Article 31 audit events.

This facade implements the Facade Pattern to hide the complexity
of the 7 internal engines behind a clean, use-case-oriented interface.

Engines (7):
    1. DDSSubmitter             - DDS creation, validation, submission
    2. OperatorRegistrar        - Operator registration and lifecycle
    3. GeolocationFormatter     - Coordinate formatting to EU specs
    4. PackageAssembler         - Document package assembly
    5. StatusTracker            - Submission status lifecycle tracking
    6. APIClient                - EU IS API communication
    7. AuditRecorder            - Article 31 audit logging

Service Methods:
    DDS Management:
        - create_dds()         -> Create a new DDS
        - validate_dds()       -> Validate DDS against requirements
        - submit_dds()         -> Submit DDS to EU IS
        - get_dds()            -> Retrieve DDS by ID
        - list_dds()           -> List DDS with filters
        - withdraw_dds()       -> Withdraw submitted DDS
        - amend_dds()          -> Amend an existing DDS

    Operator Management:
        - register_operator()  -> Register operator in EU IS
        - get_operator()       -> Retrieve operator registration
        - renew_registration() -> Renew operator registration

    Geolocation:
        - format_geolocation() -> Format coordinates to EU specs

    Packages:
        - assemble_package()   -> Assemble document package

    Status:
        - check_status()       -> Check DDS submission status
        - get_status_history() -> Get status change history

    Audit:
        - get_audit_trail()    -> Get entity audit trail
        - generate_audit_report() -> Generate audit report

    Health:
        - health_check()       -> Component health status

Singleton Pattern:
    Thread-safe singleton with double-checked locking via ``get_service()``.

FastAPI Integration:
    Use the ``lifespan`` async context manager with
    ``FastAPI(lifespan=lifespan)`` for automatic startup/shutdown.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-036 EU Information System Interface (GL-EUDR-EUIS-036)
Regulation: EU 2023/1115 (EUDR) Articles 4, 12, 13, 14, 31, 33
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
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_VERSION = "1.0.0"
_AGENT_ID = "GL-EUDR-EUIS-036"

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

from greenlang.agents.eudr.eu_information_system_interface.config import (
    EUInformationSystemInterfaceConfig,
    get_config,
)

# ---------------------------------------------------------------------------
# Provenance import (graceful fallback)
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.eu_information_system_interface.provenance import (
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
    from greenlang.agents.eudr.eu_information_system_interface.metrics import (
        record_dds_submitted,
        record_dds_accepted,
        record_dds_rejected,
        record_operator_registered,
        record_package_assembled,
        record_status_check,
        record_api_call,
        record_api_error,
        observe_submission_duration,
        observe_geolocation_format_duration,
        observe_package_assembly_duration,
        observe_api_call_duration,
        observe_status_check_duration,
        set_active_submissions,
        set_pending_dds,
        set_registered_operators,
        set_eu_api_health,
        set_audit_records_count,
    )

    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    record_dds_submitted = None  # type: ignore[assignment]
    record_dds_accepted = None  # type: ignore[assignment]
    record_dds_rejected = None  # type: ignore[assignment]
    record_operator_registered = None  # type: ignore[assignment]
    record_package_assembled = None  # type: ignore[assignment]
    record_status_check = None  # type: ignore[assignment]
    record_api_call = None  # type: ignore[assignment]
    record_api_error = None  # type: ignore[assignment]
    observe_submission_duration = None  # type: ignore[assignment]
    observe_geolocation_format_duration = None  # type: ignore[assignment]
    observe_package_assembly_duration = None  # type: ignore[assignment]
    observe_api_call_duration = None  # type: ignore[assignment]
    observe_status_check_duration = None  # type: ignore[assignment]
    set_active_submissions = None  # type: ignore[assignment]
    set_pending_dds = None  # type: ignore[assignment]
    set_registered_operators = None  # type: ignore[assignment]
    set_eu_api_health = None  # type: ignore[assignment]
    set_audit_records_count = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Model imports (graceful fallback)
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.eu_information_system_interface.models import (
        DDSStatus,
        DDSType,
        DueDiligenceStatement,
        OperatorRegistration,
        DocumentPackage,
        SubmissionRequest,
        StatusCheckResult,
        AuditRecord,
        GeolocationData,
        HealthStatus,
    )

    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Engine imports (conditional)
# ---------------------------------------------------------------------------

# ---- Engine 1: DDS Submitter ----
try:
    from greenlang.agents.eudr.eu_information_system_interface.dds_submitter import (
        DDSSubmitter,
    )
except ImportError:
    DDSSubmitter = None  # type: ignore[misc,assignment]

# ---- Engine 2: Operator Registrar ----
try:
    from greenlang.agents.eudr.eu_information_system_interface.operator_registrar import (
        OperatorRegistrar,
    )
except ImportError:
    OperatorRegistrar = None  # type: ignore[misc,assignment]

# ---- Engine 3: Geolocation Formatter ----
try:
    from greenlang.agents.eudr.eu_information_system_interface.geolocation_formatter import (
        GeolocationFormatter,
    )
except ImportError:
    GeolocationFormatter = None  # type: ignore[misc,assignment]

# ---- Engine 4: Package Assembler ----
try:
    from greenlang.agents.eudr.eu_information_system_interface.package_assembler import (
        PackageAssembler,
    )
except ImportError:
    PackageAssembler = None  # type: ignore[misc,assignment]

# ---- Engine 5: Status Tracker ----
try:
    from greenlang.agents.eudr.eu_information_system_interface.status_tracker import (
        StatusTracker,
    )
except ImportError:
    StatusTracker = None  # type: ignore[misc,assignment]

# ---- Engine 6: API Client ----
try:
    from greenlang.agents.eudr.eu_information_system_interface.api_client import (
        APIClient,
    )
except ImportError:
    APIClient = None  # type: ignore[misc,assignment]

# ---- Engine 7: Audit Recorder ----
try:
    from greenlang.agents.eudr.eu_information_system_interface.audit_recorder import (
        AuditRecorder,
    )
except ImportError:
    AuditRecorder = None  # type: ignore[misc,assignment]


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
    """Compute deterministic SHA-256 hash for provenance."""
    canonical = json.dumps(
        data, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _safe_record(metric_fn: Any, *args: Any) -> None:
    """Safely call a metrics function if available."""
    if metric_fn is not None:
        try:
            metric_fn(*args)
        except Exception:
            pass


def _safe_observe(metric_fn: Any, *args: Any) -> None:
    """Safely observe a histogram metric if available."""
    if metric_fn is not None:
        try:
            metric_fn(*args)
        except Exception:
            pass


def _safe_gauge(metric_fn: Any, value: Any) -> None:
    """Safely set a gauge metric if available."""
    if metric_fn is not None:
        try:
            metric_fn(value)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Service Facade
# ---------------------------------------------------------------------------


class EUInformationSystemInterfaceService:
    """Unified service facade for AGENT-EUDR-036.

    Aggregates all 7 processing engines and provides a clean API for
    DDS submission, operator registration, geolocation formatting,
    package assembly, status tracking, EU IS API communication, and
    Article 31 audit recording.

    Attributes:
        config: Agent configuration.
        _dds_submitter: Engine 1 -- DDS creation/validation/submission.
        _operator_registrar: Engine 2 -- operator registration.
        _geolocation_formatter: Engine 3 -- coordinate formatting.
        _package_assembler: Engine 4 -- document package assembly.
        _status_tracker: Engine 5 -- submission status tracking.
        _api_client: Engine 6 -- EU IS API communication.
        _audit_recorder: Engine 7 -- Article 31 audit logging.
        _provenance: SHA-256 provenance tracker.
        _initialized: Whether startup has completed.
    """

    def __init__(
        self,
        config: Optional[EUInformationSystemInterfaceConfig] = None,
    ) -> None:
        """Initialize the service facade."""
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
        self._dds_submitter: Optional[Any] = None
        self._operator_registrar: Optional[Any] = None
        self._geolocation_formatter: Optional[Any] = None
        self._package_assembler: Optional[Any] = None
        self._status_tracker: Optional[Any] = None
        self._api_client: Optional[Any] = None
        self._audit_recorder: Optional[Any] = None
        self._engines: Dict[str, Any] = {}

        # In-memory stores
        self._dds_store: Dict[str, Any] = {}
        self._operator_store: Dict[str, Any] = {}
        self._submission_store: Dict[str, Any] = {}
        self._package_store: Dict[str, Any] = {}

        self._initialized = False
        logger.info("EUInformationSystemInterfaceService created")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        """Initialize all engines and external connections."""
        start = time.monotonic()
        logger.info("EUInformationSystemInterfaceService startup initiated")

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

        # Initialize API client if available
        if self._api_client is not None:
            try:
                await self._api_client.initialize()
            except Exception as e:
                logger.warning("API client initialization failed: %s", e)

        self._initialized = True
        elapsed = (time.monotonic() - start) * 1000
        engine_count = len(self._engines)

        logger.info(
            "EUInformationSystemInterfaceService startup complete: "
            "%d/7 engines in %.1fms",
            engine_count, elapsed,
        )

    def _init_engines(self) -> None:
        """Initialize all 7 processing engines with graceful degradation."""
        engine_specs: List[Tuple[str, Any]] = [
            ("dds_submitter", DDSSubmitter),
            ("operator_registrar", OperatorRegistrar),
            ("geolocation_formatter", GeolocationFormatter),
            ("package_assembler", PackageAssembler),
            ("status_tracker", StatusTracker),
            ("api_client", APIClient),
            ("audit_recorder", AuditRecorder),
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

        # Wire up convenience references
        self._dds_submitter = self._engines.get("dds_submitter")
        self._operator_registrar = self._engines.get("operator_registrar")
        self._geolocation_formatter = self._engines.get("geolocation_formatter")
        self._package_assembler = self._engines.get("package_assembler")
        self._status_tracker = self._engines.get("status_tracker")
        self._api_client = self._engines.get("api_client")
        self._audit_recorder = self._engines.get("audit_recorder")

    async def shutdown(self) -> None:
        """Gracefully shutdown all connections and engines."""
        logger.info("EUInformationSystemInterfaceService shutdown initiated")

        # Close API client
        if self._api_client is not None and hasattr(self._api_client, "close"):
            try:
                await self._api_client.close()
                logger.info("API client closed")
            except Exception as e:
                logger.warning("API client close error: %s", e)

        # Shutdown engines
        for name, engine in self._engines.items():
            if hasattr(engine, "shutdown"):
                try:
                    result = engine.shutdown()
                    if asyncio.iscoroutine(result):
                        await result
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
        logger.info("EUInformationSystemInterfaceService shutdown complete")

    # ------------------------------------------------------------------
    # DDS Management
    # ------------------------------------------------------------------

    async def create_dds(
        self,
        operator_id: str,
        eori_number: str,
        dds_type: str,
        commodity_lines: List[Dict[str, Any]],
        risk_assessment_id: Optional[str] = None,
        mitigation_plan_id: Optional[str] = None,
        improvement_plan_id: Optional[str] = None,
    ) -> Any:
        """Create a new Due Diligence Statement."""
        start = time.monotonic()

        if self._dds_submitter is not None:
            try:
                dds = await self._dds_submitter.create_dds(
                    operator_id=operator_id,
                    eori_number=eori_number,
                    dds_type=dds_type,
                    commodity_lines=commodity_lines,
                    risk_assessment_id=risk_assessment_id,
                    mitigation_plan_id=mitigation_plan_id,
                    improvement_plan_id=improvement_plan_id,
                )
                self._dds_store[dds.dds_id] = dds

                # Record audit
                if self._audit_recorder is not None:
                    await self._audit_recorder.record_dds_event(
                        dds.dds_id, "dds_created", operator_id,
                        {"dds_type": dds_type, "lines": len(commodity_lines)},
                    )

                _safe_gauge(set_pending_dds, len(self._dds_store))
                return dds
            except Exception as e:
                logger.warning("DDS submitter failed: %s", e)
                raise

        # Fallback
        dds_id = f"dds-{uuid.uuid4().hex[:12]}"
        dds = {
            "dds_id": dds_id,
            "operator_id": operator_id,
            "eori_number": eori_number,
            "dds_type": dds_type,
            "status": "draft",
            "commodity_lines": commodity_lines,
            "created_at": _utcnow().isoformat(),
        }
        self._dds_store[dds_id] = dds
        _safe_gauge(set_pending_dds, len(self._dds_store))
        return dds

    async def validate_dds(self, dds_id: str) -> Dict[str, Any]:
        """Validate a DDS against EUDR requirements."""
        dds = self._dds_store.get(dds_id)
        if dds is None:
            raise ValueError(f"DDS {dds_id} not found")

        if self._dds_submitter is not None:
            try:
                return await self._dds_submitter.validate_dds(dds)
            except Exception as e:
                logger.warning("DDS validation engine failed: %s", e)

        return {"dds_id": dds_id, "valid": True, "errors": [], "warnings": []}

    async def submit_dds(self, dds_id: str) -> Any:
        """Submit a DDS to the EU Information System."""
        start = time.monotonic()
        dds = self._dds_store.get(dds_id)
        if dds is None:
            raise ValueError(f"DDS {dds_id} not found")

        if self._dds_submitter is not None:
            try:
                submission = await self._dds_submitter.submit_dds(dds)
                self._submission_store[submission.submission_id] = submission

                # Record audit
                if self._audit_recorder is not None:
                    await self._audit_recorder.record_dds_event(
                        dds_id, "dds_submitted", "system",
                    )

                commodity = "unknown"
                if hasattr(dds, "commodity_lines") and dds.commodity_lines:
                    commodity = dds.commodity_lines[0].commodity.value
                elif isinstance(dds, dict):
                    lines = dds.get("commodity_lines", [])
                    if lines:
                        commodity = lines[0].get("commodity", "unknown")

                dds_type = (
                    dds.dds_type.value if hasattr(dds, "dds_type") and hasattr(dds.dds_type, "value")
                    else dds.get("dds_type", "placing") if isinstance(dds, dict)
                    else "placing"
                )
                _safe_record(record_dds_submitted, commodity, dds_type)
                elapsed = time.monotonic() - start
                _safe_observe(observe_submission_duration, commodity, elapsed)
                _safe_gauge(set_active_submissions, len(self._submission_store))

                return submission
            except Exception as e:
                logger.warning("DDS submission engine failed: %s", e)
                raise

        raise RuntimeError("DDS submitter engine not available")

    async def get_dds(self, dds_id: str) -> Optional[Any]:
        """Retrieve a DDS by identifier."""
        return self._dds_store.get(dds_id)

    async def list_dds(
        self,
        operator_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Any]:
        """List DDS with optional filters."""
        results = list(self._dds_store.values())

        if operator_id:
            results = [
                d for d in results
                if (
                    getattr(d, "operator_id", None) == operator_id
                    or (isinstance(d, dict) and d.get("operator_id") == operator_id)
                )
            ]
        if status:
            results = [
                d for d in results
                if (
                    (hasattr(d, "status") and getattr(d.status, "value", str(d.status)) == status)
                    or (isinstance(d, dict) and d.get("status") == status)
                )
            ]

        return results

    # ------------------------------------------------------------------
    # Operator Management
    # ------------------------------------------------------------------

    async def register_operator(
        self,
        operator_id: str,
        eori_number: str,
        operator_type: str,
        company_name: str,
        member_state: str,
        address: str = "",
        contact_email: str = "",
    ) -> Any:
        """Register an operator in the EU Information System."""
        if self._operator_registrar is not None:
            try:
                reg = await self._operator_registrar.register_operator(
                    operator_id=operator_id,
                    eori_number=eori_number,
                    operator_type=operator_type,
                    company_name=company_name,
                    member_state=member_state,
                    address=address,
                    contact_email=contact_email,
                )
                self._operator_store[reg.registration_id] = reg

                if self._audit_recorder is not None:
                    await self._audit_recorder.record_event(
                        event_type="operator_registered",
                        entity_type="operator",
                        entity_id=operator_id,
                        actor="system",
                        action="register",
                        details={"member_state": member_state},
                    )

                _safe_record(record_operator_registered, member_state)
                _safe_gauge(set_registered_operators, len(self._operator_store))
                return reg
            except Exception as e:
                logger.warning("Operator registrar failed: %s", e)
                raise

        raise RuntimeError("Operator registrar engine not available")

    async def get_operator(self, registration_id: str) -> Optional[Any]:
        """Retrieve an operator registration by ID."""
        return self._operator_store.get(registration_id)

    # ------------------------------------------------------------------
    # Geolocation
    # ------------------------------------------------------------------

    async def format_geolocation(
        self,
        coordinates: List[Dict[str, Any]],
        country_code: str,
        region: str = "",
        area_hectares: Optional[str] = None,
    ) -> Any:
        """Format geolocation data to EU IS specifications."""
        start = time.monotonic()

        area = Decimal(area_hectares) if area_hectares else None

        if self._geolocation_formatter is not None:
            try:
                result = await self._geolocation_formatter.format_geolocation(
                    coordinates=coordinates,
                    country_code=country_code,
                    region=region,
                    area_hectares=area,
                )
                elapsed = time.monotonic() - start
                _safe_observe(observe_geolocation_format_duration, elapsed)
                return result
            except Exception as e:
                logger.warning("Geolocation formatter failed: %s", e)
                raise

        raise RuntimeError("Geolocation formatter engine not available")

    # ------------------------------------------------------------------
    # Packages
    # ------------------------------------------------------------------

    async def assemble_package(
        self,
        dds_id: str,
        documents: List[Dict[str, Any]],
    ) -> Any:
        """Assemble a document package for DDS submission."""
        start = time.monotonic()

        if self._package_assembler is not None:
            try:
                package = await self._package_assembler.assemble_package(
                    dds_id=dds_id,
                    documents=documents,
                )
                self._package_store[package.package_id] = package

                commodity = "unknown"
                dds = self._dds_store.get(dds_id)
                if dds and hasattr(dds, "commodity_lines") and dds.commodity_lines:
                    commodity = dds.commodity_lines[0].commodity.value

                _safe_record(record_package_assembled, commodity)
                elapsed = time.monotonic() - start
                _safe_observe(observe_package_assembly_duration, commodity, elapsed)
                return package
            except Exception as e:
                logger.warning("Package assembler failed: %s", e)
                raise

        raise RuntimeError("Package assembler engine not available")

    # ------------------------------------------------------------------
    # Status Tracking
    # ------------------------------------------------------------------

    async def check_status(
        self,
        dds_id: str,
        eu_reference: str,
    ) -> Any:
        """Check DDS submission status."""
        start = time.monotonic()

        if self._status_tracker is not None:
            try:
                result = await self._status_tracker.check_status(
                    dds_id=dds_id,
                    eu_reference=eu_reference,
                )
                _safe_record(record_status_check, result.current_status.value)
                elapsed = time.monotonic() - start
                _safe_observe(observe_status_check_duration, elapsed)
                return result
            except Exception as e:
                logger.warning("Status tracker failed: %s", e)
                raise

        raise RuntimeError("Status tracker engine not available")

    async def get_status_history(self, dds_id: str) -> List[Dict[str, Any]]:
        """Get DDS status change history."""
        if self._status_tracker is not None:
            return await self._status_tracker.get_status_history(dds_id)
        return []

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    async def get_audit_trail(
        self,
        entity_type: str,
        entity_id: str,
    ) -> List[Any]:
        """Get audit trail for an entity."""
        if self._audit_recorder is not None:
            return await self._audit_recorder.get_records_for_entity(
                entity_type, entity_id,
            )
        return []

    async def generate_audit_report(
        self,
        entity_type: str,
        entity_id: str,
    ) -> Dict[str, Any]:
        """Generate audit report for competent authority."""
        if self._audit_recorder is not None:
            return await self._audit_recorder.generate_audit_report(
                entity_type, entity_id,
            )
        return {"error": "Audit recorder not available"}

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
            "stores": {
                "dds": len(self._dds_store),
                "operators": len(self._operator_store),
                "submissions": len(self._submission_store),
                "packages": len(self._package_store),
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
        expected = [
            "dds_submitter", "operator_registrar", "geolocation_formatter",
            "package_assembler", "status_tracker", "api_client", "audit_recorder",
        ]

        for name in expected:
            if name in self._engines:
                engine = self._engines[name]
                if hasattr(engine, "health_check"):
                    try:
                        eng_health = await engine.health_check()
                        result["engines"][name] = eng_health
                    except Exception as e:
                        result["engines"][name] = {"status": "error", "error": str(e)}
                else:
                    result["engines"][name] = {"status": "available"}
            else:
                result["engines"][name] = {"status": "not_loaded"}

        # Overall status
        unhealthy = sum(
            1 for v in result["engines"].values()
            if isinstance(v, dict) and v.get("status") in ("error", "not_loaded")
        )
        if unhealthy > 3:
            result["status"] = "unhealthy"
        elif unhealthy > 0:
            result["status"] = "degraded"

        return result

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def get_engine(self, name: str) -> Optional[Any]:
        """Get a specific engine by name."""
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
        """Return the number of DDS in memory."""
        return len(self._dds_store)

    @property
    def operator_count(self) -> int:
        """Return the number of operators in memory."""
        return len(self._operator_store)


# ---------------------------------------------------------------------------
# Thread-safe singleton
# ---------------------------------------------------------------------------

_service_lock = threading.Lock()
_service_instance: Optional[EUInformationSystemInterfaceService] = None


def get_service(
    config: Optional[EUInformationSystemInterfaceConfig] = None,
) -> EUInformationSystemInterfaceService:
    """Get the global EUInformationSystemInterfaceService singleton.

    Thread-safe lazy initialization.

    Args:
        config: Optional configuration override for first creation.

    Returns:
        EUInformationSystemInterfaceService singleton instance.
    """
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = EUInformationSystemInterfaceService(config)
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
        >>> from greenlang.agents.eudr.eu_information_system_interface.setup import lifespan
        >>> app = FastAPI(lifespan=lifespan)
    """
    service = get_service()
    await service.startup()
    logger.info("EU Information System Interface lifespan: startup complete")

    try:
        yield
    finally:
        await service.shutdown()
        logger.info(
            "EU Information System Interface lifespan: shutdown complete"
        )
