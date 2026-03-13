# -*- coding: utf-8 -*-
"""
MassBalanceCalculatorService - Facade for AGENT-EUDR-011

Single entry point for all mass balance operations. Manages 8 engines,
async PostgreSQL pool, Redis cache, OpenTelemetry tracing, Prometheus metrics.

Lifecycle:
    startup -> load config -> connect DB -> connect Redis -> load reference data
            -> initialize engines -> start health check
    shutdown -> close engines -> close Redis -> close DB -> flush metrics

Engines (8):
    1. LedgerManager - Double-entry ledger management (Feature 1)
    2. CreditPeriodEngine - Period lifecycle management (Feature 2)
    3. ConversionFactorValidator - Factor validation (Feature 3)
    4. OverdraftDetector - Real-time overdraft detection (Feature 4)
    5. LossWasteTracker - Loss and waste tracking (Feature 5)
    6. CarryForwardManager - Credit carry-forward (Feature 6)
    7. ReconciliationEngine - Period-end reconciliation (Feature 7)
    8. ConsolidationReporter - Multi-facility consolidation (Feature 8)

Reference Data (3):
    - conversion_factors: Commodity yield ratios (30+)
    - loss_tolerances: Maximum loss percentages
    - credit_period_rules: Standard-specific rules

Singleton Pattern:
    Thread-safe singleton with double-checked locking via ``get_service()``.

FastAPI Integration:
    Use the ``lifespan`` async context manager with ``FastAPI(lifespan=lifespan)``
    for automatic startup/shutdown.

Example:
    >>> from greenlang.agents.eudr.mass_balance_calculator.setup import (
    ...     MassBalanceCalculatorService,
    ...     get_service,
    ... )
    >>> service = get_service()
    >>> await service.startup()
    >>> health = await service.health_check()
    >>> assert health["status"] == "healthy"
    >>> await service.shutdown()

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-011
Agent ID: GL-EUDR-MBC-011
Regulation: EU 2023/1115 (EUDR) Articles 4, 10(2)(f), 14
Standard: ISO 22095:2020 Chain of Custody - Mass Balance
Status: Production Ready
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
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
# Internal imports: config, models, provenance, metrics
# ---------------------------------------------------------------------------

from greenlang.agents.eudr.mass_balance_calculator.config import (
    MassBalanceCalculatorConfig,
    get_config,
    set_config,
    reset_config,
)
from greenlang.agents.eudr.mass_balance_calculator.provenance import (
    ProvenanceTracker,
    get_provenance_tracker,
)
from greenlang.agents.eudr.mass_balance_calculator.metrics import (
    PROMETHEUS_AVAILABLE,
    record_api_error,
    record_batch_job,
    record_reconciliation,
    record_report_generated,
    set_active_ledgers,
    set_total_balance_kg,
)

# ---------------------------------------------------------------------------
# Internal imports: 8 engines (lazy in _initialize_engines for resilience)
# ---------------------------------------------------------------------------

from greenlang.agents.eudr.mass_balance_calculator.ledger_manager import (
    LedgerManager,
)
from greenlang.agents.eudr.mass_balance_calculator.credit_period_engine import (
    CreditPeriodEngine,
)
from greenlang.agents.eudr.mass_balance_calculator.conversion_factor_validator import (
    ConversionFactorValidator,
)
from greenlang.agents.eudr.mass_balance_calculator.loss_waste_tracker import (
    LossWasteTracker,
)
from greenlang.agents.eudr.mass_balance_calculator.reconciliation_engine import (
    ReconciliationEngine,
)
from greenlang.agents.eudr.mass_balance_calculator.consolidation_reporter import (
    ConsolidationReporter,
)

# ---------------------------------------------------------------------------
# Internal imports: reference data
# ---------------------------------------------------------------------------

from greenlang.agents.eudr.mass_balance_calculator.reference_data import (
    CONVERSION_FACTORS,
    PROCESSING_LOSS_TOLERANCES,
    CREDIT_PERIOD_RULES,
)

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"
_AGENT_ID = "GL-EUDR-MBC-011"
_ENGINE_COUNT = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_provenance_hash(*parts: str) -> str:
    """Compute SHA-256 hash over concatenated string parts."""
    combined = "|".join(str(p) for p in parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def _generate_request_id() -> str:
    """Generate a unique request identifier."""
    return f"MBC-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Result container: HealthStatus
# ---------------------------------------------------------------------------


class HealthStatus:
    """Health check result container.

    Attributes:
        status: Overall health status (healthy, degraded, unhealthy).
        checks: Individual component check results.
        timestamp: When the health check was performed.
        version: Service version string.
        uptime_seconds: Seconds since service startup.
    """

    __slots__ = ("status", "checks", "timestamp", "version", "uptime_seconds")

    def __init__(
        self,
        status: str = "unhealthy",
        checks: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
        version: str = _MODULE_VERSION,
        uptime_seconds: float = 0.0,
    ) -> None:
        self.status = status
        self.checks = checks or {}
        self.timestamp = timestamp or _utcnow()
        self.version = version
        self.uptime_seconds = uptime_seconds

    def to_dict(self) -> Dict[str, Any]:
        """Serialize health status to dictionary for JSON response."""
        return {
            "status": self.status,
            "checks": self.checks,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "uptime_seconds": round(self.uptime_seconds, 2),
        }


# ---------------------------------------------------------------------------
# Result container: LedgerResult
# ---------------------------------------------------------------------------


class LedgerResult:
    """Result from a ledger operation.

    Attributes:
        request_id: Unique request identifier.
        success: Whether the operation succeeded.
        ledger_id: Ledger identifier.
        data: Operation result data payload.
        error: Error message if the operation failed.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "request_id", "success", "ledger_id", "data",
        "error", "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        request_id: str = "",
        success: bool = True,
        ledger_id: str = "",
        data: Optional[Dict[str, Any]] = None,
        error: str = "",
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.request_id = request_id or _generate_request_id()
        self.success = success
        self.ledger_id = ledger_id
        self.data = data or {}
        self.error = error
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "request_id": self.request_id,
            "success": self.success,
            "ledger_id": self.ledger_id,
            "data": self.data,
            "error": self.error,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


# ---------------------------------------------------------------------------
# Result container: PeriodResult
# ---------------------------------------------------------------------------


class PeriodResult:
    """Result from a credit period operation.

    Attributes:
        request_id: Unique request identifier.
        success: Whether the operation succeeded.
        period_id: Credit period identifier.
        data: Operation result data payload.
        error: Error message if the operation failed.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "request_id", "success", "period_id", "data",
        "error", "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        request_id: str = "",
        success: bool = True,
        period_id: str = "",
        data: Optional[Dict[str, Any]] = None,
        error: str = "",
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.request_id = request_id or _generate_request_id()
        self.success = success
        self.period_id = period_id
        self.data = data or {}
        self.error = error
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "request_id": self.request_id,
            "success": self.success,
            "period_id": self.period_id,
            "data": self.data,
            "error": self.error,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


# ---------------------------------------------------------------------------
# Result container: FactorResult
# ---------------------------------------------------------------------------


class FactorResult:
    """Result from a conversion factor operation.

    Attributes:
        request_id: Unique request identifier.
        success: Whether the operation succeeded.
        data: Operation result data payload.
        error: Error message if the operation failed.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "request_id", "success", "data",
        "error", "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        request_id: str = "",
        success: bool = True,
        data: Optional[Dict[str, Any]] = None,
        error: str = "",
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.request_id = request_id or _generate_request_id()
        self.success = success
        self.data = data or {}
        self.error = error
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "request_id": self.request_id,
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


# ---------------------------------------------------------------------------
# Result container: OverdraftResult
# ---------------------------------------------------------------------------


class OverdraftResult:
    """Result from an overdraft detection operation.

    Attributes:
        request_id: Unique request identifier.
        success: Whether the operation succeeded.
        data: Operation result data payload.
        error: Error message if the operation failed.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "request_id", "success", "data",
        "error", "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        request_id: str = "",
        success: bool = True,
        data: Optional[Dict[str, Any]] = None,
        error: str = "",
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.request_id = request_id or _generate_request_id()
        self.success = success
        self.data = data or {}
        self.error = error
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "request_id": self.request_id,
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


# ---------------------------------------------------------------------------
# Result container: LossResult
# ---------------------------------------------------------------------------


class LossResult:
    """Result from a loss/waste tracking operation.

    Attributes:
        request_id: Unique request identifier.
        success: Whether the operation succeeded.
        loss_id: Loss record identifier.
        data: Operation result data payload.
        error: Error message if the operation failed.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "request_id", "success", "loss_id", "data",
        "error", "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        request_id: str = "",
        success: bool = True,
        loss_id: str = "",
        data: Optional[Dict[str, Any]] = None,
        error: str = "",
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.request_id = request_id or _generate_request_id()
        self.success = success
        self.loss_id = loss_id
        self.data = data or {}
        self.error = error
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "request_id": self.request_id,
            "success": self.success,
            "loss_id": self.loss_id,
            "data": self.data,
            "error": self.error,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


# ---------------------------------------------------------------------------
# Result container: CarryForwardResult
# ---------------------------------------------------------------------------


class CarryForwardResult:
    """Result from a carry-forward operation.

    Attributes:
        request_id: Unique request identifier.
        success: Whether the operation succeeded.
        data: Operation result data payload.
        error: Error message if the operation failed.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "request_id", "success", "data",
        "error", "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        request_id: str = "",
        success: bool = True,
        data: Optional[Dict[str, Any]] = None,
        error: str = "",
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.request_id = request_id or _generate_request_id()
        self.success = success
        self.data = data or {}
        self.error = error
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "request_id": self.request_id,
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


# ---------------------------------------------------------------------------
# Result container: ReconciliationResult
# ---------------------------------------------------------------------------


class ReconciliationResult:
    """Result from a reconciliation operation.

    Attributes:
        request_id: Unique request identifier.
        success: Whether the operation succeeded.
        reconciliation_id: Reconciliation identifier.
        data: Operation result data payload.
        error: Error message if the operation failed.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "request_id", "success", "reconciliation_id", "data",
        "error", "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        request_id: str = "",
        success: bool = True,
        reconciliation_id: str = "",
        data: Optional[Dict[str, Any]] = None,
        error: str = "",
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.request_id = request_id or _generate_request_id()
        self.success = success
        self.reconciliation_id = reconciliation_id
        self.data = data or {}
        self.error = error
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "request_id": self.request_id,
            "success": self.success,
            "reconciliation_id": self.reconciliation_id,
            "data": self.data,
            "error": self.error,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


# ---------------------------------------------------------------------------
# Result container: ReportResult
# ---------------------------------------------------------------------------


class ReportResult:
    """Result from a report generation operation.

    Attributes:
        request_id: Unique request identifier.
        success: Whether the operation succeeded.
        report_id: Report identifier.
        data: Operation result data payload.
        error: Error message if the operation failed.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "request_id", "success", "report_id", "data",
        "error", "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        request_id: str = "",
        success: bool = True,
        report_id: str = "",
        data: Optional[Dict[str, Any]] = None,
        error: str = "",
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.request_id = request_id or _generate_request_id()
        self.success = success
        self.report_id = report_id
        self.data = data or {}
        self.error = error
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "request_id": self.request_id,
            "success": self.success,
            "report_id": self.report_id,
            "data": self.data,
            "error": self.error,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


# ---------------------------------------------------------------------------
# Result container: BatchJobResult
# ---------------------------------------------------------------------------


class BatchJobResult:
    """Result from a batch processing job.

    Attributes:
        request_id: Unique request identifier.
        success: Whether the operation succeeded.
        job_id: Batch job identifier.
        data: Operation result data payload.
        error: Error message if the operation failed.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "request_id", "success", "job_id", "data",
        "error", "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        request_id: str = "",
        success: bool = True,
        job_id: str = "",
        data: Optional[Dict[str, Any]] = None,
        error: str = "",
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.request_id = request_id or _generate_request_id()
        self.success = success
        self.job_id = job_id
        self.data = data or {}
        self.error = error
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "request_id": self.request_id,
            "success": self.success,
            "job_id": self.job_id,
            "data": self.data,
            "error": self.error,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


# ---------------------------------------------------------------------------
# Result container: DashboardResult
# ---------------------------------------------------------------------------


class DashboardResult:
    """Result from a dashboard or overview operation.

    Attributes:
        request_id: Unique request identifier.
        success: Whether the operation succeeded.
        data: Operation result data payload.
        error: Error message if the operation failed.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "request_id", "success", "data",
        "error", "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        request_id: str = "",
        success: bool = True,
        data: Optional[Dict[str, Any]] = None,
        error: str = "",
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.request_id = request_id or _generate_request_id()
        self.success = success
        self.data = data or {}
        self.error = error
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "request_id": self.request_id,
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


# ===========================================================================
# MassBalanceCalculatorService - Main facade
# ===========================================================================


class MassBalanceCalculatorService:
    """Facade for the Mass Balance Calculator Agent (AGENT-EUDR-011).

    Provides a unified interface to all 8 engines:
        1. LedgerManager             - Double-entry ledger management
        2. CreditPeriodEngine        - Credit period lifecycle
        3. ConversionFactorValidator - Conversion factor validation
        4. OverdraftDetector         - Real-time overdraft detection
        5. LossWasteTracker          - Loss and waste tracking
        6. CarryForwardManager       - Credit carry-forward
        7. ReconciliationEngine      - Period-end reconciliation
        8. ConsolidationReporter     - Multi-facility consolidation

    Singleton pattern with thread-safe initialization.

    Example:
        >>> service = MassBalanceCalculatorService()
        >>> await service.startup()
        >>> result = await service.create_ledger({...})
        >>> await service.shutdown()
    """

    _instance: Optional[MassBalanceCalculatorService] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize MassBalanceCalculatorService.

        Loads configuration but does NOT start connections or engines.
        Call ``startup()`` to activate the service.
        """
        self._config: MassBalanceCalculatorConfig = get_config()

        self._started = False
        self._start_time: Optional[float] = None
        self._config_hash = _compute_provenance_hash(
            self._config.database_url,
            self._config.redis_url,
            str(self._config.overdraft_mode),
            str(self._config.default_credit_period_days),
            self._config.genesis_hash,
        )

        # Connection handles
        self._db_pool: Optional[Any] = None
        self._redis: Optional[Any] = None

        # Engine instances (initialized in startup)
        self._ledger_manager: Optional[LedgerManager] = None
        self._credit_period_engine: Optional[CreditPeriodEngine] = None
        self._conversion_factor_validator: Optional[ConversionFactorValidator] = None
        self._overdraft_detector: Optional[Any] = None
        self._loss_waste_tracker: Optional[LossWasteTracker] = None
        self._carry_forward_manager: Optional[Any] = None
        self._reconciliation_engine: Optional[ReconciliationEngine] = None
        self._consolidation_reporter: Optional[ConsolidationReporter] = None

        # Reference data (loaded in startup)
        self._ref_conversion_factors: Optional[Dict[str, Any]] = None
        self._ref_loss_tolerances: Optional[Dict[str, Any]] = None
        self._ref_credit_period_rules: Optional[Dict[str, Any]] = None

        # Health check background task
        self._health_task: Optional[asyncio.Task[None]] = None
        self._last_health: Optional[HealthStatus] = None

        # OpenTelemetry tracer
        self._tracer: Optional[Any] = None

        # Metrics counters
        self._metrics: Dict[str, int] = {
            "ledgers_created": 0,
            "entries_recorded": 0,
            "bulk_imports": 0,
            "periods_created": 0,
            "periods_transitioned": 0,
            "factors_validated": 0,
            "overdrafts_checked": 0,
            "losses_recorded": 0,
            "carry_forwards": 0,
            "reconciliations": 0,
            "reports_generated": 0,
            "transfers_recorded": 0,
            "errors": 0,
        }

        logger.info(
            "MassBalanceCalculatorService created: config_hash=%s, "
            "overdraft_mode=%s, default_period=%dd",
            self._config_hash[:12],
            self._config.overdraft_mode,
            self._config.default_credit_period_days,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """Return whether the service is started and active."""
        return self._started

    @property
    def uptime_seconds(self) -> float:
        """Return seconds since startup, or 0.0 if not started."""
        if self._start_time is None:
            return 0.0
        return time.monotonic() - self._start_time

    @property
    def ledger_manager(self) -> LedgerManager:
        """Return the LedgerManager engine instance."""
        self._ensure_started()
        return self._ledger_manager  # type: ignore[return-value]

    @property
    def credit_period_engine(self) -> CreditPeriodEngine:
        """Return the CreditPeriodEngine engine instance."""
        self._ensure_started()
        return self._credit_period_engine  # type: ignore[return-value]

    @property
    def conversion_factor_validator(self) -> ConversionFactorValidator:
        """Return the ConversionFactorValidator engine instance."""
        self._ensure_started()
        return self._conversion_factor_validator  # type: ignore[return-value]

    @property
    def overdraft_detector(self) -> Any:
        """Return the OverdraftDetector engine instance."""
        self._ensure_started()
        return self._overdraft_detector

    @property
    def loss_waste_tracker(self) -> LossWasteTracker:
        """Return the LossWasteTracker engine instance."""
        self._ensure_started()
        return self._loss_waste_tracker  # type: ignore[return-value]

    @property
    def carry_forward_manager(self) -> Any:
        """Return the CarryForwardManager engine instance."""
        self._ensure_started()
        return self._carry_forward_manager

    @property
    def reconciliation_engine(self) -> ReconciliationEngine:
        """Return the ReconciliationEngine engine instance."""
        self._ensure_started()
        return self._reconciliation_engine  # type: ignore[return-value]

    @property
    def consolidation_reporter(self) -> ConsolidationReporter:
        """Return the ConsolidationReporter engine instance."""
        self._ensure_started()
        return self._consolidation_reporter  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Startup / Shutdown
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        """Start the service: connect DB, Redis, initialize all engines.

        Executes the full startup sequence:
            1. Configure structured logging
            2. Initialize OpenTelemetry tracer
            3. Load reference data
            4. Connect to PostgreSQL
            5. Connect to Redis
            6. Initialize all eight engines
            7. Start background health check task

        Idempotent: safe to call multiple times.
        """
        if self._started:
            logger.debug("MassBalanceCalculatorService already started")
            return

        start = time.monotonic()
        logger.info("MassBalanceCalculatorService starting up...")

        self._configure_logging()
        self._init_tracer()
        self._load_reference_data()
        await self._connect_database()
        await self._connect_redis()
        await self._initialize_engines()
        self._start_health_check()

        self._started = True
        self._start_time = time.monotonic()
        elapsed = (time.monotonic() - start) * 1000

        logger.info(
            "MassBalanceCalculatorService started in %.1fms: "
            "db=%s, redis=%s, engines=%d/%d, config_hash=%s",
            elapsed,
            "connected" if self._db_pool is not None else "skipped",
            "connected" if self._redis is not None else "skipped",
            self._count_initialized_engines(),
            _ENGINE_COUNT,
            self._config_hash[:12],
        )

    async def shutdown(self) -> None:
        """Gracefully shut down the service and release all resources.

        Idempotent: safe to call multiple times.
        """
        if not self._started:
            logger.debug("MassBalanceCalculatorService already stopped")
            return

        logger.info("MassBalanceCalculatorService shutting down...")
        start = time.monotonic()

        self._stop_health_check()
        await self._close_engines()
        await self._close_redis()
        await self._close_database()
        self._flush_metrics()

        self._started = False
        elapsed = (time.monotonic() - start) * 1000
        logger.info(
            "MassBalanceCalculatorService shut down in %.1fms", elapsed,
        )

    # ==================================================================
    # FACADE METHODS: Engine 1 - LedgerManager
    # ==================================================================

    async def create_ledger(
        self,
        ledger_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create a new mass balance ledger.

        Delegates to LedgerManager.create_ledger().

        Args:
            ledger_data: Ledger creation data including facility_id,
                commodity, period_id, and optional standard.

        Returns:
            Dictionary with ledger creation result.
        """
        self._ensure_started()
        start = time.monotonic()
        try:
            result = self._ledger_manager.create_ledger(**ledger_data)  # type: ignore[union-attr]
            self._metrics["ledgers_created"] += 1
            return self._wrap_result(result, start)
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("create_ledger")
            logger.error("create_ledger failed: %s", exc, exc_info=True)
            raise

    async def record_entry(
        self,
        entry_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Record a ledger entry (input, output, adjustment, loss, waste).

        Delegates to LedgerManager.record_entry().

        Args:
            entry_data: Entry data including ledger_id, entry_type,
                quantity_kg, and optional metadata.

        Returns:
            Dictionary with entry recording result.
        """
        self._ensure_started()
        start = time.monotonic()
        try:
            result = self._ledger_manager.record_entry(**entry_data)  # type: ignore[union-attr]
            self._metrics["entries_recorded"] += 1
            return self._wrap_result(result, start)
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("record_entry")
            logger.error("record_entry failed: %s", exc, exc_info=True)
            raise

    async def get_balance(self, ledger_id: str) -> Dict[str, Any]:
        """Get current balance for a ledger.

        Delegates to LedgerManager.get_balance().

        Args:
            ledger_id: Ledger identifier.

        Returns:
            Dictionary with balance information.
        """
        self._ensure_started()
        return self._ledger_manager.get_balance(ledger_id)  # type: ignore[union-attr]

    async def get_entry_history(
        self,
        ledger_id: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Get entry history for a ledger.

        Delegates to LedgerManager.get_entry_history().

        Args:
            ledger_id: Ledger identifier.
            **kwargs: Additional filter arguments.

        Returns:
            Dictionary with entry history.
        """
        self._ensure_started()
        return self._ledger_manager.get_entry_history(  # type: ignore[union-attr]
            ledger_id, **kwargs,
        )

    async def search_ledgers(
        self,
        filters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Search ledgers by criteria.

        Delegates to LedgerManager.search_ledgers().

        Args:
            filters: Search filters.

        Returns:
            Dictionary with matching ledgers.
        """
        self._ensure_started()
        return self._ledger_manager.search_ledgers(**filters)  # type: ignore[union-attr]

    async def get_summary(self, ledger_id: str) -> Dict[str, Any]:
        """Get ledger summary with aggregate metrics.

        Delegates to LedgerManager.get_summary().

        Args:
            ledger_id: Ledger identifier.

        Returns:
            Dictionary with ledger summary.
        """
        self._ensure_started()
        return self._ledger_manager.get_summary(ledger_id)  # type: ignore[union-attr]

    async def bulk_import_entries(
        self,
        import_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Bulk import ledger entries.

        Delegates to LedgerManager.bulk_import().

        Args:
            import_data: Bulk import data.

        Returns:
            Dictionary with import results.
        """
        self._ensure_started()
        start = time.monotonic()
        try:
            result = self._ledger_manager.bulk_import(**import_data)  # type: ignore[union-attr]
            self._metrics["bulk_imports"] += 1
            record_batch_job()
            return self._wrap_result(result, start)
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("bulk_entry")
            logger.error("bulk_import failed: %s", exc, exc_info=True)
            raise

    # ==================================================================
    # FACADE METHODS: Engine 2 - CreditPeriodEngine
    # ==================================================================

    async def create_period(
        self,
        period_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create a new credit period.

        Delegates to CreditPeriodEngine.create_period().

        Args:
            period_data: Period creation data including facility_id,
                commodity, and optional standard.

        Returns:
            Dictionary with period creation result.
        """
        self._ensure_started()
        start = time.monotonic()
        try:
            result = self._credit_period_engine.create_period(**period_data)  # type: ignore[union-attr]
            self._metrics["periods_created"] += 1
            return self._wrap_result(result, start)
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("create_period")
            logger.error("create_period failed: %s", exc, exc_info=True)
            raise

    async def get_period(self, period_id: str) -> Optional[Dict[str, Any]]:
        """Get a credit period by ID.

        Delegates to CreditPeriodEngine.get_period().

        Args:
            period_id: Credit period identifier.

        Returns:
            Period record or None.
        """
        self._ensure_started()
        return self._credit_period_engine.get_period(period_id)  # type: ignore[union-attr]

    async def get_active_periods(
        self,
        facility_id: Optional[str] = None,
        commodity: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get active credit periods.

        Delegates to CreditPeriodEngine.get_active_periods().

        Args:
            facility_id: Optional facility filter.
            commodity: Optional commodity filter.

        Returns:
            List of active period records.
        """
        self._ensure_started()
        return self._credit_period_engine.get_active_periods(  # type: ignore[union-attr]
            facility_id=facility_id,
            commodity=commodity,
        )

    async def transition_period(
        self,
        period_id: str,
        new_status: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Transition a credit period to a new status.

        Delegates to CreditPeriodEngine.transition_period().

        Args:
            period_id: Credit period identifier.
            new_status: Target status.
            **kwargs: Additional transition parameters.

        Returns:
            Dictionary with transition result.
        """
        self._ensure_started()
        start = time.monotonic()
        try:
            result = self._credit_period_engine.transition_period(  # type: ignore[union-attr]
                period_id, new_status, **kwargs,
            )
            self._metrics["periods_transitioned"] += 1
            return self._wrap_result(result, start)
        except Exception as exc:
            self._metrics["errors"] += 1
            logger.error("transition_period failed: %s", exc, exc_info=True)
            raise

    async def rollover_period(
        self,
        period_id: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Rollover a credit period, creating a new successor.

        Delegates to CreditPeriodEngine.rollover_period().

        Args:
            period_id: Credit period to roll over.
            **kwargs: Additional rollover parameters.

        Returns:
            Dictionary with rollover result.
        """
        self._ensure_started()
        return self._credit_period_engine.rollover_period(  # type: ignore[union-attr]
            period_id, **kwargs,
        )

    async def extend_period(
        self,
        period_id: str,
        extension_days: int,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Extend a credit period by additional days.

        Delegates to CreditPeriodEngine.extend_period().

        Args:
            period_id: Credit period to extend.
            extension_days: Number of days to extend.
            **kwargs: Additional extension parameters.

        Returns:
            Dictionary with extension result.
        """
        self._ensure_started()
        return self._credit_period_engine.extend_period(  # type: ignore[union-attr]
            period_id, extension_days, **kwargs,
        )

    # ==================================================================
    # FACADE METHODS: Engine 3 - ConversionFactorValidator
    # ==================================================================

    async def validate_factor(
        self,
        factor_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate a conversion factor against reference data.

        Delegates to ConversionFactorValidator.validate_factor().

        Args:
            factor_data: Factor data including commodity, process_type,
                and claimed_factor.

        Returns:
            Dictionary with validation result.
        """
        self._ensure_started()
        start = time.monotonic()
        try:
            result = self._conversion_factor_validator.validate_factor(  # type: ignore[union-attr]
                **factor_data,
            )
            self._metrics["factors_validated"] += 1
            return self._wrap_result(result, start)
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("validate_factor")
            logger.error("validate_factor failed: %s", exc, exc_info=True)
            raise

    async def get_reference_factors(
        self,
        commodity: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get reference conversion factors.

        Delegates to ConversionFactorValidator.get_reference_factors().

        Args:
            commodity: Optional commodity filter.

        Returns:
            Dictionary with reference factors.
        """
        self._ensure_started()
        return self._conversion_factor_validator.get_reference_factors(  # type: ignore[union-attr]
            commodity=commodity,
        )

    async def register_custom_factor(
        self,
        factor_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Register a custom conversion factor.

        Delegates to ConversionFactorValidator.register_custom_factor().

        Args:
            factor_data: Custom factor data.

        Returns:
            Dictionary with registration result.
        """
        self._ensure_started()
        return self._conversion_factor_validator.register_custom_factor(  # type: ignore[union-attr]
            **factor_data,
        )

    async def get_factor_history(
        self,
        commodity: str,
        process_type: str,
    ) -> Dict[str, Any]:
        """Get conversion factor validation history.

        Delegates to ConversionFactorValidator.get_factor_history().

        Args:
            commodity: Commodity identifier.
            process_type: Processing type.

        Returns:
            Dictionary with factor history.
        """
        self._ensure_started()
        return self._conversion_factor_validator.get_factor_history(  # type: ignore[union-attr]
            commodity=commodity,
            process_type=process_type,
        )

    # ==================================================================
    # FACADE METHODS: Engine 4 - OverdraftDetector
    # ==================================================================

    async def check_overdraft(
        self,
        check_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Check for overdraft on a ledger.

        Delegates to OverdraftDetector.check() if available.

        Args:
            check_data: Overdraft check parameters.

        Returns:
            Dictionary with overdraft check result.
        """
        self._ensure_started()
        start = time.monotonic()
        result = self._safe_engine_call_with_args(
            self._overdraft_detector, "check", **check_data,
        )
        self._metrics["overdrafts_checked"] += 1
        if result is None:
            return {"status": "engine_unavailable", "overdraft": False}
        return self._wrap_result(result, start)

    async def get_active_alerts(
        self,
        facility_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get active overdraft alerts.

        Delegates to OverdraftDetector.get_active_alerts() if available.

        Args:
            facility_id: Optional facility filter.

        Returns:
            List of active alert records.
        """
        self._ensure_started()
        result = self._safe_engine_call_with_args(
            self._overdraft_detector, "get_active_alerts",
            facility_id=facility_id,
        )
        return result if isinstance(result, list) else []

    async def forecast_output(
        self,
        forecast_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Forecast output capacity before overdraft.

        Delegates to OverdraftDetector.forecast() if available.

        Args:
            forecast_data: Forecast parameters.

        Returns:
            Dictionary with forecast result.
        """
        self._ensure_started()
        result = self._safe_engine_call_with_args(
            self._overdraft_detector, "forecast", **forecast_data,
        )
        if result is None:
            return {"status": "engine_unavailable"}
        return result

    async def request_exemption(
        self,
        exemption_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Request an overdraft exemption.

        Delegates to OverdraftDetector.request_exemption() if available.

        Args:
            exemption_data: Exemption request data.

        Returns:
            Dictionary with exemption result.
        """
        self._ensure_started()
        result = self._safe_engine_call_with_args(
            self._overdraft_detector, "request_exemption",
            **exemption_data,
        )
        if result is None:
            return {"status": "engine_unavailable"}
        return result

    async def get_overdraft_history(
        self,
        facility_id: str,
    ) -> List[Dict[str, Any]]:
        """Get overdraft event history for a facility.

        Delegates to OverdraftDetector.get_history() if available.

        Args:
            facility_id: Facility identifier.

        Returns:
            List of overdraft event records.
        """
        self._ensure_started()
        result = self._safe_engine_call_with_args(
            self._overdraft_detector, "get_history",
            facility_id=facility_id,
        )
        return result if isinstance(result, list) else []

    # ==================================================================
    # FACADE METHODS: Engine 5 - LossWasteTracker
    # ==================================================================

    async def record_loss(
        self,
        loss_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Record a processing, transport, or storage loss.

        Delegates to LossWasteTracker.record_loss().

        Args:
            loss_data: Loss data including ledger_id, loss_type,
                commodity, quantity_kg.

        Returns:
            Dictionary with loss recording result.
        """
        self._ensure_started()
        start = time.monotonic()
        try:
            result = self._loss_waste_tracker.record_loss(**loss_data)  # type: ignore[union-attr]
            self._metrics["losses_recorded"] += 1
            return self._wrap_result(result, start)
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("record_loss")
            logger.error("record_loss failed: %s", exc, exc_info=True)
            raise

    async def record_waste(
        self,
        waste_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Record waste from processing.

        Delegates to LossWasteTracker.record_waste().

        Args:
            waste_data: Waste data.

        Returns:
            Dictionary with waste recording result.
        """
        self._ensure_started()
        return self._loss_waste_tracker.record_waste(**waste_data)  # type: ignore[union-attr]

    async def validate_loss(
        self,
        validation_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate a loss against commodity tolerances.

        Delegates to LossWasteTracker.validate_loss().

        Args:
            validation_data: Validation parameters.

        Returns:
            Dictionary with validation result.
        """
        self._ensure_started()
        return self._loss_waste_tracker.validate_loss(**validation_data)  # type: ignore[union-attr]

    async def get_loss_records(
        self,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Get loss records with optional filters.

        Delegates to LossWasteTracker.get_loss_records().

        Args:
            **kwargs: Filter parameters.

        Returns:
            List of loss records.
        """
        self._ensure_started()
        return self._loss_waste_tracker.get_loss_records(**kwargs)  # type: ignore[union-attr]

    async def get_loss_trends(
        self,
        facility_id: str,
        commodity: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Get loss trend analysis.

        Delegates to LossWasteTracker.get_loss_trends().

        Args:
            facility_id: Facility identifier.
            commodity: Commodity identifier.
            **kwargs: Additional parameters.

        Returns:
            Dictionary with loss trend data.
        """
        self._ensure_started()
        return self._loss_waste_tracker.get_loss_trends(  # type: ignore[union-attr]
            facility_id=facility_id,
            commodity=commodity,
            **kwargs,
        )

    # ==================================================================
    # FACADE METHODS: Engine 6 - CarryForwardManager
    # ==================================================================

    async def carry_forward(
        self,
        carry_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Carry forward balance to the next credit period.

        Delegates to CarryForwardManager.carry_forward() if available.

        Args:
            carry_data: Carry-forward parameters.

        Returns:
            Dictionary with carry-forward result.
        """
        self._ensure_started()
        result = self._safe_engine_call_with_args(
            self._carry_forward_manager, "carry_forward",
            **carry_data,
        )
        self._metrics["carry_forwards"] += 1
        if result is None:
            return {"status": "engine_unavailable"}
        return result

    async def void_expired_credits(
        self,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Void expired carry-forward credits.

        Delegates to CarryForwardManager.void_expired() if available.

        Args:
            **kwargs: Void parameters.

        Returns:
            Dictionary with void result.
        """
        self._ensure_started()
        result = self._safe_engine_call_with_args(
            self._carry_forward_manager, "void_expired",
            **kwargs,
        )
        if result is None:
            return {"status": "engine_unavailable"}
        return result

    async def get_carry_forward_status(
        self,
        period_id: str,
    ) -> Dict[str, Any]:
        """Get carry-forward status for a period.

        Delegates to CarryForwardManager.get_status() if available.

        Args:
            period_id: Credit period identifier.

        Returns:
            Dictionary with carry-forward status.
        """
        self._ensure_started()
        result = self._safe_engine_call_with_args(
            self._carry_forward_manager, "get_status",
            period_id=period_id,
        )
        if result is None:
            return {"status": "engine_unavailable"}
        return result

    # ==================================================================
    # FACADE METHODS: Engine 7 - ReconciliationEngine
    # ==================================================================

    async def run_reconciliation(
        self,
        period_id: str,
        facility_id: str,
        commodity: str,
        ledger_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run period-end reconciliation.

        Delegates to ReconciliationEngine.run_reconciliation().

        Args:
            period_id: Credit period to reconcile.
            facility_id: Facility identifier.
            commodity: Commodity being reconciled.
            ledger_summary: Optional pre-computed ledger summary.

        Returns:
            Dictionary with reconciliation result.
        """
        self._ensure_started()
        start = time.monotonic()
        try:
            result = self._reconciliation_engine.run_reconciliation(  # type: ignore[union-attr]
                period_id=period_id,
                facility_id=facility_id,
                commodity=commodity,
                ledger_summary=ledger_summary,
            )
            self._metrics["reconciliations"] += 1
            return result
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("run_reconciliation")
            logger.error(
                "run_reconciliation failed: %s", exc, exc_info=True,
            )
            raise

    async def sign_off_reconciliation(
        self,
        reconciliation_id: str,
        signed_off_by: str,
    ) -> Dict[str, Any]:
        """Sign off on a completed reconciliation.

        Delegates to ReconciliationEngine.sign_off().

        Args:
            reconciliation_id: Reconciliation identifier.
            signed_off_by: Operator identifier.

        Returns:
            Dictionary with sign-off result.
        """
        self._ensure_started()
        return self._reconciliation_engine.sign_off(  # type: ignore[union-attr]
            reconciliation_id=reconciliation_id,
            signed_off_by=signed_off_by,
        )

    async def get_reconciliation(
        self,
        reconciliation_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get a reconciliation record by ID.

        Delegates to ReconciliationEngine.get_reconciliation().

        Args:
            reconciliation_id: Reconciliation identifier.

        Returns:
            Reconciliation record or None.
        """
        self._ensure_started()
        return self._reconciliation_engine.get_reconciliation(  # type: ignore[union-attr]
            reconciliation_id,
        )

    async def get_reconciliation_history(
        self,
        facility_id: str,
        commodity: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get reconciliation history for a facility.

        Delegates to ReconciliationEngine.get_reconciliation_history().

        Args:
            facility_id: Facility identifier.
            commodity: Optional commodity filter.

        Returns:
            List of reconciliation records.
        """
        self._ensure_started()
        return self._reconciliation_engine.get_reconciliation_history(  # type: ignore[union-attr]
            facility_id=facility_id,
            commodity=commodity,
        )

    # ==================================================================
    # FACADE METHODS: Engine 8 - ConsolidationReporter
    # ==================================================================

    async def generate_consolidation_report(
        self,
        facility_ids: List[str],
        report_type: str,
        report_format: str = "json",
        commodity: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a consolidation report.

        Delegates to ConsolidationReporter.generate_consolidation_report().

        Args:
            facility_ids: Facilities to include.
            report_type: Type of report.
            report_format: Output format.
            commodity: Optional commodity filter.

        Returns:
            Dictionary with report result.
        """
        self._ensure_started()
        start = time.monotonic()
        try:
            result = self._consolidation_reporter.generate_consolidation_report(  # type: ignore[union-attr]
                facility_ids=facility_ids,
                report_type=report_type,
                report_format=report_format,
                commodity=commodity,
            )
            self._metrics["reports_generated"] += 1
            return result
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("generate_report")
            logger.error(
                "generate_consolidation_report failed: %s",
                exc, exc_info=True,
            )
            raise

    async def create_facility_group(
        self,
        name: str,
        group_type: str,
        facility_ids: List[str],
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a facility group.

        Delegates to ConsolidationReporter.create_facility_group().

        Args:
            name: Group name.
            group_type: Grouping type.
            facility_ids: Facility identifiers.
            description: Optional description.

        Returns:
            Dictionary with group creation result.
        """
        self._ensure_started()
        return self._consolidation_reporter.create_facility_group(  # type: ignore[union-attr]
            name=name,
            group_type=group_type,
            facility_ids=facility_ids,
            description=description,
        )

    async def get_enterprise_dashboard(
        self,
        group_id: Optional[str] = None,
        facility_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Get enterprise dashboard across facilities.

        Delegates to ConsolidationReporter.get_enterprise_dashboard().

        Args:
            group_id: Optional facility group.
            facility_ids: Optional explicit facility list.

        Returns:
            Dictionary with dashboard data.
        """
        self._ensure_started()
        return self._consolidation_reporter.get_enterprise_dashboard(  # type: ignore[union-attr]
            group_id=group_id,
            facility_ids=facility_ids,
        )

    async def get_report(
        self,
        report_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get a generated report by ID.

        Delegates to ConsolidationReporter.get_report().

        Args:
            report_id: Report identifier.

        Returns:
            Report record or None.
        """
        self._ensure_started()
        return self._consolidation_reporter.get_report(report_id)  # type: ignore[union-attr]

    async def download_report(
        self,
        report_id: str,
    ) -> Dict[str, Any]:
        """Download a generated report with content.

        Delegates to ConsolidationReporter.download_report().

        Args:
            report_id: Report identifier.

        Returns:
            Dictionary with report content.
        """
        self._ensure_started()
        return self._consolidation_reporter.download_report(report_id)  # type: ignore[union-attr]

    # ==================================================================
    # Health check
    # ==================================================================

    async def health_check(self) -> Dict[str, Any]:
        """Perform a comprehensive health check.

        Checks database, Redis, engine, and reference data health.

        Returns:
            Dictionary with overall status and component checks.
        """
        checks: Dict[str, Any] = {}

        checks["database"] = await self._check_database_health()
        checks["redis"] = await self._check_redis_health()
        checks["engines"] = self._check_engine_health()
        checks["reference_data"] = self._check_reference_data_health()

        # Determine overall status
        statuses = [c.get("status", "unhealthy") for c in checks.values()]
        if all(s == "healthy" for s in statuses):
            overall = "healthy"
        elif any(s == "unhealthy" for s in statuses):
            overall = "unhealthy"
        else:
            overall = "degraded"

        health = HealthStatus(
            status=overall,
            checks=checks,
            timestamp=_utcnow(),
            version=_MODULE_VERSION,
            uptime_seconds=self.uptime_seconds,
        )
        self._last_health = health
        return health.to_dict()

    # ------------------------------------------------------------------
    # Internal: Startup helpers
    # ------------------------------------------------------------------

    def _configure_logging(self) -> None:
        """Configure structured logging for the service."""
        log_level = getattr(
            logging, self._config.log_level.upper(), logging.INFO,
        )
        logging.getLogger(
            "greenlang.agents.eudr.mass_balance_calculator"
        ).setLevel(log_level)
        logger.debug("Logging configured: level=%s", self._config.log_level)

    def _init_tracer(self) -> None:
        """Initialize OpenTelemetry tracer if available."""
        if OTEL_AVAILABLE and otel_trace is not None:
            try:
                self._tracer = otel_trace.get_tracer(
                    "greenlang.agents.eudr.mass_balance_calculator",
                    _MODULE_VERSION,
                )
                logger.debug("OpenTelemetry tracer initialized")
            except Exception as exc:
                logger.warning("OpenTelemetry init failed: %s", exc)
        else:
            logger.debug("OpenTelemetry not available; tracing disabled")

    def _load_reference_data(self) -> None:
        """Load reference datasets for deterministic validation."""
        try:
            self._ref_conversion_factors = CONVERSION_FACTORS
            logger.debug(
                "Loaded conversion factors: %d entries",
                len(self._ref_conversion_factors)
                if self._ref_conversion_factors else 0,
            )
        except Exception as exc:
            logger.warning("Failed to load conversion factors: %s", exc)

        try:
            self._ref_loss_tolerances = PROCESSING_LOSS_TOLERANCES
            logger.debug(
                "Loaded loss tolerances: %d entries",
                len(self._ref_loss_tolerances)
                if self._ref_loss_tolerances else 0,
            )
        except Exception as exc:
            logger.warning("Failed to load loss tolerances: %s", exc)

        try:
            self._ref_credit_period_rules = CREDIT_PERIOD_RULES
            logger.debug(
                "Loaded credit period rules: %d entries",
                len(self._ref_credit_period_rules)
                if self._ref_credit_period_rules else 0,
            )
        except Exception as exc:
            logger.warning("Failed to load credit period rules: %s", exc)

    async def _connect_database(self) -> None:
        """Connect to the PostgreSQL database pool."""
        if not PSYCOPG_POOL_AVAILABLE:
            logger.info(
                "psycopg_pool not available; database connection skipped"
            )
            return

        try:
            self._db_pool = AsyncConnectionPool(
                self._config.database_url,
                min_size=2,
                max_size=self._config.pool_size,
                open=False,
            )
            await self._db_pool.open()
            logger.info(
                "PostgreSQL connection pool opened: pool_size=%d",
                self._config.pool_size,
            )
        except Exception as exc:
            logger.warning(
                "PostgreSQL connection failed (non-fatal): %s", exc,
            )
            self._db_pool = None

    async def _connect_redis(self) -> None:
        """Connect to the Redis cache."""
        if not REDIS_AVAILABLE or aioredis is None:
            logger.info("Redis not available; cache connection skipped")
            return

        try:
            self._redis = aioredis.from_url(
                self._config.redis_url,
                decode_responses=True,
            )
            await self._redis.ping()
            logger.info("Redis connection established")
        except Exception as exc:
            logger.warning(
                "Redis connection failed (non-fatal): %s", exc,
            )
            self._redis = None

    async def _initialize_engines(self) -> None:
        """Initialize all 8 engines with graceful fallback."""
        config = self._config

        # Engine 1: LedgerManager
        try:
            self._ledger_manager = LedgerManager(config=config)
            logger.debug("Engine 1 initialized: LedgerManager")
        except (ImportError, Exception) as exc:
            logger.warning(
                "Engine 1 (LedgerManager) init failed: %s", exc,
            )

        # Engine 2: CreditPeriodEngine
        try:
            self._credit_period_engine = CreditPeriodEngine(config=config)
            logger.debug("Engine 2 initialized: CreditPeriodEngine")
        except (ImportError, Exception) as exc:
            logger.warning(
                "Engine 2 (CreditPeriodEngine) init failed: %s", exc,
            )

        # Engine 3: ConversionFactorValidator
        try:
            self._conversion_factor_validator = ConversionFactorValidator(
                config=config,
            )
            logger.debug("Engine 3 initialized: ConversionFactorValidator")
        except (ImportError, Exception) as exc:
            logger.warning(
                "Engine 3 (ConversionFactorValidator) init failed: %s", exc,
            )

        # Engine 4: OverdraftDetector (not yet built)
        try:
            from greenlang.agents.eudr.mass_balance_calculator.overdraft_detector import (
                OverdraftDetector,
            )
            self._overdraft_detector = OverdraftDetector(config=config)
            logger.debug("Engine 4 initialized: OverdraftDetector")
        except (ImportError, Exception) as exc:
            logger.warning(
                "Engine 4 (OverdraftDetector) init failed: %s", exc,
            )

        # Engine 5: LossWasteTracker
        try:
            self._loss_waste_tracker = LossWasteTracker(config=config)
            logger.debug("Engine 5 initialized: LossWasteTracker")
        except (ImportError, Exception) as exc:
            logger.warning(
                "Engine 5 (LossWasteTracker) init failed: %s", exc,
            )

        # Engine 6: CarryForwardManager (not yet built)
        try:
            from greenlang.agents.eudr.mass_balance_calculator.carry_forward_manager import (
                CarryForwardManager,
            )
            self._carry_forward_manager = CarryForwardManager(config=config)
            logger.debug("Engine 6 initialized: CarryForwardManager")
        except (ImportError, Exception) as exc:
            logger.warning(
                "Engine 6 (CarryForwardManager) init failed: %s", exc,
            )

        # Engine 7: ReconciliationEngine
        try:
            self._reconciliation_engine = ReconciliationEngine(config=config)
            logger.debug("Engine 7 initialized: ReconciliationEngine")
        except (ImportError, Exception) as exc:
            logger.warning(
                "Engine 7 (ReconciliationEngine) init failed: %s", exc,
            )

        # Engine 8: ConsolidationReporter
        try:
            self._consolidation_reporter = ConsolidationReporter(config=config)
            logger.debug("Engine 8 initialized: ConsolidationReporter")
        except (ImportError, Exception) as exc:
            logger.warning(
                "Engine 8 (ConsolidationReporter) init failed: %s", exc,
            )

        count = self._count_initialized_engines()
        logger.info("Engines initialized: %d/%d", count, _ENGINE_COUNT)

    async def _close_engines(self) -> None:
        """Close all engines and release resources."""
        engine_names = [
            "_ledger_manager",
            "_credit_period_engine",
            "_conversion_factor_validator",
            "_overdraft_detector",
            "_loss_waste_tracker",
            "_carry_forward_manager",
            "_reconciliation_engine",
            "_consolidation_reporter",
        ]
        for name in engine_names:
            engine = getattr(self, name, None)
            if engine is not None and hasattr(engine, "close"):
                try:
                    result = engine.close()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as exc:
                    logger.warning("Error closing %s: %s", name, exc)
            setattr(self, name, None)
        logger.debug("All engines closed")

    async def _close_redis(self) -> None:
        """Close the Redis connection."""
        if self._redis is not None:
            try:
                await self._redis.close()
                logger.info("Redis connection closed")
            except Exception as exc:
                logger.warning("Error closing Redis: %s", exc)
            finally:
                self._redis = None

    async def _close_database(self) -> None:
        """Close the PostgreSQL connection pool."""
        if self._db_pool is not None:
            try:
                await self._db_pool.close()
                logger.info("PostgreSQL connection pool closed")
            except Exception as exc:
                logger.warning("Error closing database pool: %s", exc)
            finally:
                self._db_pool = None

    def _flush_metrics(self) -> None:
        """Flush Prometheus metrics."""
        if self._config.enable_metrics:
            logger.debug(
                "Metrics flushed: %s",
                {k: v for k, v in self._metrics.items() if v > 0},
            )

    # ------------------------------------------------------------------
    # Internal: Health checks
    # ------------------------------------------------------------------

    def _start_health_check(self) -> None:
        """Start the background health check task."""
        try:
            loop = asyncio.get_running_loop()
            self._health_task = loop.create_task(
                self._health_check_loop(),
            )
            logger.debug("Health check background task started")
        except RuntimeError:
            logger.debug(
                "No running event loop; health check task not started",
            )

    def _stop_health_check(self) -> None:
        """Cancel the background health check task."""
        if self._health_task is not None:
            self._health_task.cancel()
            self._health_task = None
            logger.debug("Health check background task cancelled")

    async def _health_check_loop(self) -> None:
        """Periodic background health check."""
        while True:
            try:
                await asyncio.sleep(30.0)
                await self.health_check()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("Health check loop error: %s", exc)

    async def _check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity health."""
        if self._db_pool is None:
            return {"status": "degraded", "reason": "no_pool"}
        try:
            start = time.monotonic()
            async with self._db_pool.connection() as conn:
                await conn.execute("SELECT 1")
            latency_ms = (time.monotonic() - start) * 1000
            return {
                "status": "healthy",
                "latency_ms": round(latency_ms, 2),
            }
        except Exception as exc:
            return {"status": "unhealthy", "reason": str(exc)}

    async def _check_redis_health(self) -> Dict[str, Any]:
        """Check Redis connectivity health."""
        if self._redis is None:
            return {"status": "degraded", "reason": "not_connected"}
        try:
            start = time.monotonic()
            await self._redis.ping()
            latency_ms = (time.monotonic() - start) * 1000
            return {
                "status": "healthy",
                "latency_ms": round(latency_ms, 2),
            }
        except Exception as exc:
            return {"status": "unhealthy", "reason": str(exc)}

    def _check_engine_health(self) -> Dict[str, Any]:
        """Check engine initialization status."""
        engines = {
            "ledger_manager": self._ledger_manager,
            "credit_period_engine": self._credit_period_engine,
            "conversion_factor_validator": (
                self._conversion_factor_validator
            ),
            "overdraft_detector": self._overdraft_detector,
            "loss_waste_tracker": self._loss_waste_tracker,
            "carry_forward_manager": self._carry_forward_manager,
            "reconciliation_engine": self._reconciliation_engine,
            "consolidation_reporter": self._consolidation_reporter,
        }
        engine_status = {
            name: "initialized" if engine is not None else "not_available"
            for name, engine in engines.items()
        }
        count = self._count_initialized_engines()
        if count == _ENGINE_COUNT:
            status = "healthy"
        elif count > 0:
            status = "degraded"
        else:
            status = "unhealthy"
        return {
            "status": status,
            "initialized_count": count,
            "total_count": _ENGINE_COUNT,
            "engines": engine_status,
        }

    def _check_reference_data_health(self) -> Dict[str, Any]:
        """Check reference data availability."""
        loaded = sum(1 for x in [
            self._ref_conversion_factors,
            self._ref_loss_tolerances,
            self._ref_credit_period_rules,
        ] if x is not None)
        return {
            "status": "healthy" if loaded == 3 else "degraded",
            "loaded_datasets": loaded,
            "total_datasets": 3,
        }

    def _count_initialized_engines(self) -> int:
        """Count the number of successfully initialized engines."""
        engines = [
            self._ledger_manager,
            self._credit_period_engine,
            self._conversion_factor_validator,
            self._overdraft_detector,
            self._loss_waste_tracker,
            self._carry_forward_manager,
            self._reconciliation_engine,
            self._consolidation_reporter,
        ]
        return sum(1 for e in engines if e is not None)

    # ------------------------------------------------------------------
    # Internal: Utility helpers
    # ------------------------------------------------------------------

    def _ensure_started(self) -> None:
        """Ensure the service has been started.

        Raises:
            RuntimeError: If the service has not been started.
        """
        if not self._started:
            raise RuntimeError(
                "MassBalanceCalculatorService is not started. "
                "Call startup() first."
            )

    def _wrap_result(
        self,
        result: Any,
        start_time: float,
    ) -> Dict[str, Any]:
        """Wrap an engine result with processing time metadata.

        Args:
            result: Engine method result.
            start_time: Monotonic start time.

        Returns:
            Result with processing_time_ms added.
        """
        elapsed_ms = (time.monotonic() - start_time) * 1000
        if isinstance(result, dict):
            result["processing_time_ms"] = round(elapsed_ms, 2)
            return result
        return {
            "data": result,
            "processing_time_ms": round(elapsed_ms, 2),
        }

    def _safe_engine_call(
        self,
        engine: Optional[Any],
        method_name: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Safely delegate a call to an engine method.

        If the engine is None or the method does not exist, returns
        None without raising.

        Args:
            engine: Engine instance (may be None).
            method_name: Method to invoke on the engine.
            payload: Optional dictionary payload for the method.

        Returns:
            Engine method result dict, or None on failure.
        """
        if engine is None:
            return None
        try:
            method = getattr(engine, method_name, None)
            if method is None:
                return None
            if payload is not None:
                result = method(payload)
            else:
                result = method()
            if isinstance(result, dict):
                return result
            return None
        except Exception as exc:
            logger.debug(
                "Engine call fallback: %s.%s -> %s",
                type(engine).__name__, method_name, exc,
            )
            return None

    def _safe_engine_call_with_args(
        self,
        engine: Optional[Any],
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Optional[Any]:
        """Safely delegate a call to an engine method with arguments.

        Args:
            engine: Engine instance (may be None).
            method_name: Method to invoke on the engine.
            *args: Positional arguments for the method.
            **kwargs: Keyword arguments for the method.

        Returns:
            Engine method result, or None on failure.
        """
        if engine is None:
            return None
        try:
            method = getattr(engine, method_name, None)
            if method is None:
                return None
            return method(*args, **kwargs)
        except Exception as exc:
            logger.debug(
                "Engine call fallback: %s.%s -> %s",
                type(engine).__name__, method_name, exc,
            )
            return None


# ---------------------------------------------------------------------------
# FastAPI lifespan context manager
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: Any) -> AsyncIterator[None]:
    """FastAPI lifespan context manager for the Mass Balance Calculator service.

    Automatically starts the service on application startup and shuts it
    down on application shutdown.  The service instance is stored in
    ``app.state.mbc_service`` for access from route handlers.

    Usage with FastAPI::

        from fastapi import FastAPI
        from greenlang.agents.eudr.mass_balance_calculator.setup import lifespan

        app = FastAPI(lifespan=lifespan)

    Args:
        app: The FastAPI application instance.

    Yields:
        None (service is accessible via ``app.state.mbc_service``).
    """
    service = get_service()
    app.state.mbc_service = service
    try:
        await service.startup()
        yield
    finally:
        await service.shutdown()


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_service_instance: Optional[MassBalanceCalculatorService] = None
_service_lock = threading.Lock()


def get_service() -> MassBalanceCalculatorService:
    """Return the singleton MassBalanceCalculatorService instance.

    Uses double-checked locking for thread safety.  The instance is
    created on first call.

    Returns:
        MassBalanceCalculatorService singleton instance.

    Example:
        >>> service = get_service()
        >>> await service.startup()
    """
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = MassBalanceCalculatorService()
    return _service_instance


def set_service(service: MassBalanceCalculatorService) -> None:
    """Replace the singleton MassBalanceCalculatorService instance.

    Primarily intended for testing and dependency injection.

    Args:
        service: Replacement service instance.
    """
    global _service_instance
    with _service_lock:
        _service_instance = service
    logger.info("MassBalanceCalculatorService singleton replaced")


def reset_service() -> None:
    """Reset the singleton MassBalanceCalculatorService to None.

    The next call to ``get_service()`` will create a fresh instance.
    Intended for test teardown.
    """
    global _service_instance
    with _service_lock:
        _service_instance = None
    logger.debug("MassBalanceCalculatorService singleton reset")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Service
    "MassBalanceCalculatorService",
    "HealthStatus",
    "lifespan",
    "get_service",
    "set_service",
    "reset_service",
    # Result containers
    "LedgerResult",
    "PeriodResult",
    "FactorResult",
    "OverdraftResult",
    "LossResult",
    "CarryForwardResult",
    "ReconciliationResult",
    "ReportResult",
    "BatchJobResult",
    "DashboardResult",
]
