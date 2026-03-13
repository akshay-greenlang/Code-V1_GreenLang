# -*- coding: utf-8 -*-
"""
Authority Communication Manager Service Facade - AGENT-EUDR-040

High-level service facade that wires together all 7 processing engines
into a single cohesive entry point. Provides the primary API used by
the FastAPI router layer to manage authority communications, handle
information requests, coordinate inspections, process non-compliance
notices, manage appeals, exchange documents, route notifications,
and render multi-language templates per EUDR Articles 15-19, 31.

Engines (7):
    1. RequestHandler           - Process information requests (Article 17)
    2. InspectionCoordinator    - Schedule and manage inspections (Article 15)
    3. NonComplianceManager     - Handle violations and penalties (Article 16)
    4. AppealProcessor          - Manage administrative appeals (Article 19)
    5. DocumentExchange         - Secure document sharing with encryption
    6. NotificationRouter       - Multi-channel notification delivery
    7. TemplateEngine           - Multi-language template rendering

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-040 Authority Communication Manager (GL-EUDR-ACM-040)
Regulation: EU 2023/1115 (EUDR) Articles 15, 16, 17, 19, 31
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
from decimal import Decimal
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_VERSION = "1.0.0"
_AGENT_ID = "GL-EUDR-ACM-040"

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

from greenlang.agents.eudr.authority_communication_manager.config import (
    AuthorityCommunicationManagerConfig,
    EU_MEMBER_STATES,
    get_config,
)

# ---------------------------------------------------------------------------
# Provenance import (graceful fallback)
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.authority_communication_manager.provenance import (
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
    from greenlang.agents.eudr.authority_communication_manager.metrics import (
        record_communication_created,
        record_communication_sent,
        record_communication_responded,
        record_information_request_received,
        record_inspection_scheduled,
        record_non_compliance_issued,
        record_appeal_filed,
        record_document_exchanged,
        record_notification_sent,
        record_deadline_reminder_sent,
        record_api_error,
        observe_processing_duration,
        observe_communication_creation,
        set_pending_communications,
        set_overdue_responses,
        set_active_appeals,
        set_pending_inspections,
        set_open_non_compliance,
        set_template_count,
        set_authority_count,
        set_member_states_active,
    )

    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    # Set all metric functions to None for safe fallback
    record_communication_created = None  # type: ignore[assignment]
    record_communication_sent = None  # type: ignore[assignment]
    record_communication_responded = None  # type: ignore[assignment]
    record_information_request_received = None  # type: ignore[assignment]
    record_inspection_scheduled = None  # type: ignore[assignment]
    record_non_compliance_issued = None  # type: ignore[assignment]
    record_appeal_filed = None  # type: ignore[assignment]
    record_document_exchanged = None  # type: ignore[assignment]
    record_notification_sent = None  # type: ignore[assignment]
    record_deadline_reminder_sent = None  # type: ignore[assignment]
    record_api_error = None  # type: ignore[assignment]
    observe_processing_duration = None  # type: ignore[assignment]
    observe_communication_creation = None  # type: ignore[assignment]
    set_pending_communications = None  # type: ignore[assignment]
    set_overdue_responses = None  # type: ignore[assignment]
    set_active_appeals = None  # type: ignore[assignment]
    set_pending_inspections = None  # type: ignore[assignment]
    set_open_non_compliance = None  # type: ignore[assignment]
    set_template_count = None  # type: ignore[assignment]
    set_authority_count = None  # type: ignore[assignment]
    set_member_states_active = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Model imports (graceful fallback)
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.authority_communication_manager.models import (
        Communication,
        CommunicationPriority,
        CommunicationStatus,
        CommunicationType,
        HealthStatus,
        LanguageCode,
    )

    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Engine imports (conditional)
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.authority_communication_manager.request_handler import (
        RequestHandler,
    )
except ImportError:
    RequestHandler = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.eudr.authority_communication_manager.inspection_coordinator import (
        InspectionCoordinator,
    )
except ImportError:
    InspectionCoordinator = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.eudr.authority_communication_manager.non_compliance_manager import (
        NonComplianceManager,
    )
except ImportError:
    NonComplianceManager = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.eudr.authority_communication_manager.appeal_processor import (
        AppealProcessor,
    )
except ImportError:
    AppealProcessor = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.eudr.authority_communication_manager.document_exchange import (
        DocumentExchange,
    )
except ImportError:
    DocumentExchange = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.eudr.authority_communication_manager.notification_router import (
        NotificationRouter,
    )
except ImportError:
    NotificationRouter = None  # type: ignore[misc,assignment]

try:
    from greenlang.agents.eudr.authority_communication_manager.template_engine import (
        TemplateEngine,
    )
except ImportError:
    TemplateEngine = None  # type: ignore[misc,assignment]


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


class AuthorityCommunicationManagerService:
    """Unified service facade for AGENT-EUDR-040.

    Aggregates all 7 processing engines and provides a clean API for
    authority communication management per EUDR Articles 15-19, 31.

    Attributes:
        config: Agent configuration.
        _request_handler: Engine 1 -- information request processing.
        _inspection_coordinator: Engine 2 -- inspection coordination.
        _non_compliance_manager: Engine 3 -- violation management.
        _appeal_processor: Engine 4 -- appeal processing.
        _document_exchange: Engine 5 -- secure document sharing.
        _notification_router: Engine 6 -- notification delivery.
        _template_engine: Engine 7 -- template rendering.
        _initialized: Whether startup has completed.

    Example:
        >>> service = AuthorityCommunicationManagerService()
        >>> await service.startup()
        >>> health = await service.health_check()
        >>> assert health["status"] == "healthy"
    """

    def __init__(
        self,
        config: Optional[AuthorityCommunicationManagerConfig] = None,
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

        # Engine references (lazy initialized in startup)
        self._request_handler: Optional[Any] = None
        self._inspection_coordinator: Optional[Any] = None
        self._non_compliance_manager: Optional[Any] = None
        self._appeal_processor: Optional[Any] = None
        self._document_exchange: Optional[Any] = None
        self._notification_router: Optional[Any] = None
        self._template_engine: Optional[Any] = None
        self._engines: Dict[str, Any] = {}

        # In-memory stores
        self._communications: Dict[str, Any] = {}
        self._threads: Dict[str, Any] = {}
        self._reminders: Dict[str, Any] = {}
        self._approval_workflows: Dict[str, Any] = {}

        self._start_time: Optional[float] = None
        self._initialized = False

        logger.info("AuthorityCommunicationManagerService created")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        """Initialize all engines and external connections."""
        start = time.monotonic()
        self._start_time = start
        logger.info("AuthorityCommunicationManagerService startup initiated")

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

        # Initialize Redis (message queue)
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
                logger.info("Redis connection established (message queue ready)")
            except Exception as e:
                logger.warning(f"Redis init failed: {e}")
                self._redis = None

        # Initialize engines
        self._init_engines()

        # Load default templates
        if self._template_engine is not None:
            try:
                count = self._template_engine.load_default_templates()
                _safe_gauge(set_template_count, count)
                logger.info(f"Template library loaded: {count} templates")
            except Exception as e:
                logger.warning(f"Template library load failed: {e}")

        # Set authority count gauge
        _safe_gauge(set_authority_count, len(EU_MEMBER_STATES))
        _safe_gauge(set_member_states_active, len(EU_MEMBER_STATES))

        self._initialized = True
        elapsed = (time.monotonic() - start) * 1000
        engine_count = len(self._engines)

        logger.info(
            "AuthorityCommunicationManagerService startup complete: "
            "%d/7 engines in %.1fms",
            engine_count,
            elapsed,
        )

    def _init_engines(self) -> None:
        """Initialize all 7 processing engines with graceful degradation."""
        engine_specs: List[Tuple[str, Any]] = [
            ("request_handler", RequestHandler),
            ("inspection_coordinator", InspectionCoordinator),
            ("non_compliance_manager", NonComplianceManager),
            ("appeal_processor", AppealProcessor),
            ("document_exchange", DocumentExchange),
            ("notification_router", NotificationRouter),
            ("template_engine", TemplateEngine),
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
        self._request_handler = self._engines.get("request_handler")
        self._inspection_coordinator = self._engines.get("inspection_coordinator")
        self._non_compliance_manager = self._engines.get("non_compliance_manager")
        self._appeal_processor = self._engines.get("appeal_processor")
        self._document_exchange = self._engines.get("document_exchange")
        self._notification_router = self._engines.get("notification_router")
        self._template_engine = self._engines.get("template_engine")

    async def shutdown(self) -> None:
        """Gracefully shutdown all connections and engines."""
        logger.info("AuthorityCommunicationManagerService shutdown initiated")

        for name, engine in self._engines.items():
            if hasattr(engine, "shutdown"):
                try:
                    result = engine.shutdown()
                    if asyncio.iscoroutine(result):
                        await result
                    logger.info(f"Engine '{name}' shut down")
                except Exception as e:
                    logger.warning(f"Engine '{name}' shutdown error: {e}")

        if self._redis is not None:
            try:
                await self._redis.close()
                logger.info("Redis connection closed")
            except Exception as e:
                logger.warning(f"Redis close error: {e}")

        if self._db_pool is not None:
            try:
                await self._db_pool.close()
                logger.info("PostgreSQL pool closed")
            except Exception as e:
                logger.warning(f"PostgreSQL pool close error: {e}")

        self._initialized = False
        logger.info("AuthorityCommunicationManagerService shutdown complete")

    # ------------------------------------------------------------------
    # Communication Management
    # ------------------------------------------------------------------

    async def create_communication(
        self,
        operator_id: str,
        authority_id: str,
        member_state: str,
        communication_type: str,
        subject: str,
        body: str = "",
        priority: str = "normal",
        language: str = "en",
        dds_reference: str = "",
        document_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a new communication between operator and authority.

        Args:
            operator_id: Operator identifier.
            authority_id: Authority identifier.
            member_state: ISO 3166-1 alpha-2 member state code.
            communication_type: Type of communication.
            subject: Communication subject.
            body: Communication body text.
            priority: Priority level.
            language: Communication language.
            dds_reference: Related DDS reference.
            document_ids: Attached document IDs.

        Returns:
            Communication data dictionary.
        """
        start = time.monotonic()
        communication_id = _new_uuid()
        now = _utcnow()

        # Calculate deadline based on priority
        deadline = self._calculate_deadline(priority)

        communication = {
            "communication_id": communication_id,
            "operator_id": operator_id,
            "authority_id": authority_id,
            "member_state": member_state,
            "communication_type": communication_type,
            "status": "pending",
            "priority": priority,
            "subject": subject,
            "body": body,
            "language": language,
            "dds_reference": dds_reference,
            "deadline": deadline.isoformat() if deadline else None,
            "document_ids": document_ids or [],
            "created_at": now.isoformat(),
            "provenance_hash": _compute_hash({
                "communication_id": communication_id,
                "operator_id": operator_id,
                "authority_id": authority_id,
                "communication_type": communication_type,
                "created_at": now.isoformat(),
            }),
        }

        self._communications[communication_id] = communication
        _safe_record(record_communication_created, communication_type, member_state)
        _safe_gauge(set_pending_communications, len(self._communications))

        elapsed = time.monotonic() - start
        logger.info(
            "Communication %s created (type=%s, state=%s) in %.1fms",
            communication_id,
            communication_type,
            member_state,
            elapsed * 1000,
        )

        return communication

    async def get_communication(
        self,
        communication_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a communication by identifier."""
        return self._communications.get(communication_id)

    async def respond_to_communication(
        self,
        communication_id: str,
        responder_id: str,
        body: str,
        document_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Submit a response to a communication.

        Args:
            communication_id: Communication to respond to.
            responder_id: Identity of responder.
            body: Response body text.
            document_ids: Supporting document IDs.

        Returns:
            Updated communication data.

        Raises:
            ValueError: If communication not found.
        """
        comm = self._communications.get(communication_id)
        if comm is None:
            raise ValueError(f"Communication {communication_id} not found")

        now = _utcnow()
        comm["status"] = "responded"
        comm["responded_at"] = now.isoformat()
        comm["response_body"] = body
        comm["response_by"] = responder_id
        comm["response_documents"] = document_ids or []

        _safe_record(
            record_communication_responded,
            comm.get("communication_type", "unknown"),
        )

        logger.info(
            "Communication %s responded to by %s",
            communication_id,
            responder_id,
        )
        return comm

    async def list_pending_communications(
        self,
        operator_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List pending communications."""
        results = [
            c for c in self._communications.values()
            if c.get("status") == "pending"
        ]
        if operator_id:
            results = [
                c for c in results
                if c.get("operator_id") == operator_id
            ]
        return results

    async def list_overdue_communications(self) -> List[Dict[str, Any]]:
        """List overdue communications past their deadline."""
        now = _utcnow()
        results = []
        for c in self._communications.values():
            if c.get("status") not in ("responded", "closed", "archived"):
                deadline_str = c.get("deadline")
                if deadline_str:
                    deadline = datetime.fromisoformat(deadline_str)
                    if deadline < now:
                        results.append(c)
        return results

    # ------------------------------------------------------------------
    # Delegated operations to engines
    # ------------------------------------------------------------------

    async def handle_information_request(self, **kwargs: Any) -> Any:
        """Delegate to RequestHandler engine."""
        if self._request_handler is not None:
            result = await self._request_handler.receive_request(**kwargs)
            _safe_record(
                record_information_request_received,
                kwargs.get("request_type", "unknown"),
            )
            return result
        raise RuntimeError("RequestHandler engine not available")

    async def schedule_inspection(self, **kwargs: Any) -> Any:
        """Delegate to InspectionCoordinator engine."""
        if self._inspection_coordinator is not None:
            result = await self._inspection_coordinator.schedule_inspection(
                **kwargs
            )
            _safe_record(
                record_inspection_scheduled,
                kwargs.get("inspection_type", "unknown"),
            )
            return result
        raise RuntimeError("InspectionCoordinator engine not available")

    async def record_violation(self, **kwargs: Any) -> Any:
        """Delegate to NonComplianceManager engine."""
        if self._non_compliance_manager is not None:
            result = await self._non_compliance_manager.record_violation(
                **kwargs
            )
            _safe_record(
                record_non_compliance_issued,
                kwargs.get("violation_type", "unknown"),
                kwargs.get("severity", "unknown"),
            )
            return result
        raise RuntimeError("NonComplianceManager engine not available")

    async def file_appeal(self, **kwargs: Any) -> Any:
        """Delegate to AppealProcessor engine."""
        if self._appeal_processor is not None:
            result = await self._appeal_processor.file_appeal(**kwargs)
            _safe_record(
                record_appeal_filed,
                kwargs.get("authority_id", "unknown"),
            )
            return result
        raise RuntimeError("AppealProcessor engine not available")

    async def upload_document(self, **kwargs: Any) -> Any:
        """Delegate to DocumentExchange engine."""
        if self._document_exchange is not None:
            result = await self._document_exchange.upload_document(**kwargs)
            _safe_record(
                record_document_exchanged,
                kwargs.get("document_type", "unknown"),
                "upload",
            )
            return result
        raise RuntimeError("DocumentExchange engine not available")

    async def download_document(
        self,
        document_id: str,
        requestor: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Delegate to DocumentExchange engine."""
        if self._document_exchange is not None:
            return await self._document_exchange.download_document(
                document_id=document_id,
                requestor=requestor,
            )
        raise RuntimeError("DocumentExchange engine not available")

    async def send_notification(self, **kwargs: Any) -> Any:
        """Delegate to NotificationRouter engine."""
        if self._notification_router is not None:
            result = await self._notification_router.send_notification(**kwargs)
            _safe_record(
                record_notification_sent,
                kwargs.get("channel", "unknown"),
            )
            return result
        raise RuntimeError("NotificationRouter engine not available")

    async def render_template(self, **kwargs: Any) -> Dict[str, str]:
        """Delegate to TemplateEngine."""
        if self._template_engine is not None:
            return await self._template_engine.render_template(**kwargs)
        raise RuntimeError("TemplateEngine engine not available")

    async def list_templates(self, **kwargs: Any) -> List[Any]:
        """Delegate to TemplateEngine."""
        if self._template_engine is not None:
            return await self._template_engine.list_templates(**kwargs)
        return []

    async def get_authorities(
        self,
        member_state: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List configured authorities by member state.

        Args:
            member_state: Optional ISO 3166-1 alpha-2 filter.

        Returns:
            List of authority configuration dictionaries.
        """
        results = []
        for code, info in EU_MEMBER_STATES.items():
            if member_state and code != member_state.upper():
                continue
            results.append({
                "member_state": code,
                **info,
            })
        return results

    # ------------------------------------------------------------------
    # Deadline tracking
    # ------------------------------------------------------------------

    def _calculate_deadline(self, priority: str) -> Optional[datetime]:
        """Calculate deadline based on priority."""
        now = _utcnow()
        if priority == "urgent":
            return now + timedelta(hours=self.config.deadline_urgent_hours)
        elif priority == "high":
            return now + timedelta(hours=72)
        elif priority == "normal":
            return now + timedelta(days=self.config.deadline_normal_days)
        elif priority == "low":
            return now + timedelta(days=10)
        elif priority == "routine":
            return now + timedelta(days=self.config.deadline_routine_days)
        return now + timedelta(days=self.config.deadline_normal_days)

    # ------------------------------------------------------------------
    # Health check
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
                "communications": len(self._communications),
                "threads": len(self._threads),
                "reminders": len(self._reminders),
            },
            "member_states": len(EU_MEMBER_STATES),
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
            "request_handler",
            "inspection_coordinator",
            "non_compliance_manager",
            "appeal_processor",
            "document_exchange",
            "notification_router",
            "template_engine",
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
        unhealthy = sum(
            1
            for v in result["engines"].values()
            if isinstance(v, dict)
            and v.get("status") in ("error", "not_loaded")
        )
        if unhealthy > 3:
            result["status"] = "unhealthy"
        elif unhealthy > 0:
            result["status"] = "degraded"

        return result

    # ------------------------------------------------------------------
    # Component accessors
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
    def communication_count(self) -> int:
        """Return the number of communications in memory."""
        return len(self._communications)


# ---------------------------------------------------------------------------
# Thread-safe singleton
# ---------------------------------------------------------------------------

_service_lock = threading.Lock()
_service_instance: Optional[AuthorityCommunicationManagerService] = None


def get_service(
    config: Optional[AuthorityCommunicationManagerConfig] = None,
) -> AuthorityCommunicationManagerService:
    """Get the global AuthorityCommunicationManagerService singleton.

    Thread-safe lazy initialization.

    Args:
        config: Optional configuration override for first creation.

    Returns:
        AuthorityCommunicationManagerService singleton instance.
    """
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = AuthorityCommunicationManagerService(config)
    return _service_instance


def reset_service() -> None:
    """Reset the global service singleton (for testing only)."""
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
        >>> from greenlang.agents.eudr.authority_communication_manager.setup import lifespan
        >>> app = FastAPI(lifespan=lifespan)

    Args:
        app: FastAPI application instance.

    Yields:
        None -- application runs between startup and shutdown.
    """
    service = get_service()
    await service.startup()
    logger.info("Authority Communication Manager lifespan: startup complete")

    try:
        yield
    finally:
        await service.shutdown()
        logger.info(
            "Authority Communication Manager lifespan: shutdown complete"
        )
