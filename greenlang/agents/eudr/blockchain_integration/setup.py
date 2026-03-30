# -*- coding: utf-8 -*-
"""
BlockchainIntegrationService - Facade for AGENT-EUDR-013

Single entry point for all blockchain integration operations.  Manages 8
engines, async PostgreSQL pool, Redis cache, OpenTelemetry tracing,
Prometheus metrics.

Lifecycle:
    startup -> load config -> connect DB -> connect Redis -> load reference data
            -> initialize engines -> start health check
    shutdown -> close engines -> close Redis -> close DB -> flush metrics

Engines (8):
    1. TransactionAnchor          - On-chain anchoring of Merkle roots (Feature 1)
    2. SmartContractManager       - Contract deployment and lifecycle (Feature 2)
    3. MultiChainConnector        - Multi-chain connection management (Feature 3)
    4. VerificationEngine         - Anchor verification via proofs (Feature 4)
    5. EventListener              - On-chain event indexing (Feature 5)
    6. MerkleProofGenerator       - Merkle tree and proof generation (Feature 6)
    7. CrossPartySharing          - Cross-party data access grants (Feature 7)
    8. ComplianceEvidencePackager - Evidence package generation (Feature 8)

Reference Data (3):
    - chain_configs: Network configurations for Ethereum, Polygon, Fabric, Besu
    - contract_abis: Solidity ABIs for AnchorRegistry, CustodyTransfer,
      ComplianceCheck contracts
    - anchor_rules: 8 event type anchoring rules with priority levels,
      batch eligibility, required fields, gas estimates, retention rules

Singleton Pattern:
    Thread-safe singleton with double-checked locking via ``get_service()``.

FastAPI Integration:
    Use the ``lifespan`` async context manager with ``FastAPI(lifespan=lifespan)``
    for automatic startup/shutdown.

Example:
    >>> from greenlang.agents.eudr.blockchain_integration.setup import (
    ...     BlockchainIntegrationService,
    ...     get_service,
    ... )
    >>> service = get_service()
    >>> await service.startup()
    >>> health = await service.health_check()
    >>> assert health["status"] == "healthy"
    >>> await service.shutdown()

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-013
Agent ID: GL-EUDR-BCI-013
Regulation: EU 2023/1115 (EUDR) Articles 4, 10, 14
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
# Internal imports: config, provenance, metrics
# ---------------------------------------------------------------------------

from greenlang.agents.eudr.blockchain_integration.config import (
    BlockchainIntegrationConfig,
    get_config,
    set_config,
    reset_config,
)
from greenlang.agents.eudr.blockchain_integration.provenance import (
    ProvenanceTracker,
    get_provenance_tracker,
)
from greenlang.agents.eudr.blockchain_integration.metrics import (
    PROMETHEUS_AVAILABLE,
    record_anchor_created,
    record_anchor_confirmed,
    record_anchor_failed,
    record_verification,
    record_verification_tampered,
    record_merkle_tree_built,
    record_merkle_proof_generated,
    record_event_indexed,
    record_contract_deployed,
    record_access_grant,
    record_evidence_package,
    record_gas_spent,
    record_api_error,
    observe_anchor_duration,
    observe_verification_duration,
    observe_merkle_build_duration,
    set_active_listeners,
    set_pending_anchors,
)

# ---------------------------------------------------------------------------
# Internal imports: reference data
# ---------------------------------------------------------------------------

from greenlang.agents.eudr.blockchain_integration.reference_data import (
    ANCHOR_RULES,
    CHAIN_CONFIGS,
    CONTRACT_TYPES,
    GAS_ESTIMATES,
    RETENTION_RULES,
    SUPPORTED_NETWORKS,
    get_anchor_rule,
    get_chain_config,
    get_confirmation_depth,
    get_required_fields,
    validate_anchor_request,
)

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"
_AGENT_ID = "GL-EUDR-BCI-013"
_ENGINE_COUNT = 8

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_provenance_hash(*parts: str) -> str:
    """Compute SHA-256 hash over concatenated string parts."""
    combined = "|".join(str(p) for p in parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()

def _generate_request_id() -> str:
    """Generate a unique request identifier."""
    return f"BCI-{uuid.uuid4().hex[:12]}"

# ---------------------------------------------------------------------------
# Result container: HealthResult
# ---------------------------------------------------------------------------

class HealthResult:
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
        self.timestamp = timestamp or utcnow()
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
# Result container: AnchorResult
# ---------------------------------------------------------------------------

class AnchorResult:
    """Result from an anchoring operation.

    Attributes:
        request_id: Unique request identifier.
        success: Whether the operation succeeded.
        anchor_id: Anchor record identifier.
        data: Anchor result data payload.
        error: Error message if the operation failed.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "request_id", "success", "anchor_id", "data",
        "error", "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        request_id: str = "",
        success: bool = True,
        anchor_id: str = "",
        data: Optional[Dict[str, Any]] = None,
        error: str = "",
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.request_id = request_id or _generate_request_id()
        self.success = success
        self.anchor_id = anchor_id
        self.data = data or {}
        self.error = error
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "request_id": self.request_id,
            "success": self.success,
            "anchor_id": self.anchor_id,
            "data": self.data,
            "error": self.error,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }

# ---------------------------------------------------------------------------
# Result container: ContractResult
# ---------------------------------------------------------------------------

class ContractResult:
    """Result from a smart contract operation.

    Attributes:
        request_id: Unique request identifier.
        success: Whether the operation succeeded.
        contract_id: Contract identifier.
        data: Contract result data payload.
        error: Error message if the operation failed.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "request_id", "success", "contract_id", "data",
        "error", "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        request_id: str = "",
        success: bool = True,
        contract_id: str = "",
        data: Optional[Dict[str, Any]] = None,
        error: str = "",
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.request_id = request_id or _generate_request_id()
        self.success = success
        self.contract_id = contract_id
        self.data = data or {}
        self.error = error
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "request_id": self.request_id,
            "success": self.success,
            "contract_id": self.contract_id,
            "data": self.data,
            "error": self.error,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }

# ---------------------------------------------------------------------------
# Result container: ChainResult
# ---------------------------------------------------------------------------

class ChainResult:
    """Result from a chain connection operation.

    Attributes:
        request_id: Unique request identifier.
        success: Whether the operation succeeded.
        chain_id: Blockchain network identifier.
        data: Chain result data payload.
        error: Error message if the operation failed.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "request_id", "success", "chain_id", "data",
        "error", "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        request_id: str = "",
        success: bool = True,
        chain_id: str = "",
        data: Optional[Dict[str, Any]] = None,
        error: str = "",
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.request_id = request_id or _generate_request_id()
        self.success = success
        self.chain_id = chain_id
        self.data = data or {}
        self.error = error
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "request_id": self.request_id,
            "success": self.success,
            "chain_id": self.chain_id,
            "data": self.data,
            "error": self.error,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }

# ---------------------------------------------------------------------------
# Result container: VerifyResult
# ---------------------------------------------------------------------------

class VerifyResult:
    """Result from a verification operation.

    Attributes:
        request_id: Unique request identifier.
        success: Whether the operation succeeded.
        data: Verification result data payload.
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
# Result container: EventResult
# ---------------------------------------------------------------------------

class EventResult:
    """Result from an event listener operation.

    Attributes:
        request_id: Unique request identifier.
        success: Whether the operation succeeded.
        data: Event result data payload.
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
# Result container: MerkleResult
# ---------------------------------------------------------------------------

class MerkleResult:
    """Result from a Merkle tree or proof operation.

    Attributes:
        request_id: Unique request identifier.
        success: Whether the operation succeeded.
        tree_id: Merkle tree identifier.
        data: Merkle result data payload.
        error: Error message if the operation failed.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "request_id", "success", "tree_id", "data",
        "error", "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        request_id: str = "",
        success: bool = True,
        tree_id: str = "",
        data: Optional[Dict[str, Any]] = None,
        error: str = "",
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.request_id = request_id or _generate_request_id()
        self.success = success
        self.tree_id = tree_id
        self.data = data or {}
        self.error = error
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "request_id": self.request_id,
            "success": self.success,
            "tree_id": self.tree_id,
            "data": self.data,
            "error": self.error,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }

# ---------------------------------------------------------------------------
# Result container: SharingResult
# ---------------------------------------------------------------------------

class SharingResult:
    """Result from a cross-party sharing operation.

    Attributes:
        request_id: Unique request identifier.
        success: Whether the operation succeeded.
        grant_id: Access grant identifier.
        data: Sharing result data payload.
        error: Error message if the operation failed.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "request_id", "success", "grant_id", "data",
        "error", "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        request_id: str = "",
        success: bool = True,
        grant_id: str = "",
        data: Optional[Dict[str, Any]] = None,
        error: str = "",
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.request_id = request_id or _generate_request_id()
        self.success = success
        self.grant_id = grant_id
        self.data = data or {}
        self.error = error
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "request_id": self.request_id,
            "success": self.success,
            "grant_id": self.grant_id,
            "data": self.data,
            "error": self.error,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }

# ---------------------------------------------------------------------------
# Result container: EvidenceResult
# ---------------------------------------------------------------------------

class EvidenceResult:
    """Result from an evidence package operation.

    Attributes:
        request_id: Unique request identifier.
        success: Whether the operation succeeded.
        package_id: Evidence package identifier.
        data: Evidence result data payload.
        error: Error message if the operation failed.
        provenance_hash: SHA-256 hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
    """

    __slots__ = (
        "request_id", "success", "package_id", "data",
        "error", "provenance_hash", "processing_time_ms",
    )

    def __init__(
        self,
        request_id: str = "",
        success: bool = True,
        package_id: str = "",
        data: Optional[Dict[str, Any]] = None,
        error: str = "",
        provenance_hash: str = "",
        processing_time_ms: float = 0.0,
    ) -> None:
        self.request_id = request_id or _generate_request_id()
        self.success = success
        self.package_id = package_id
        self.data = data or {}
        self.error = error
        self.provenance_hash = provenance_hash
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON response."""
        return {
            "request_id": self.request_id,
            "success": self.success,
            "package_id": self.package_id,
            "data": self.data,
            "error": self.error,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }

# ---------------------------------------------------------------------------
# Result container: BatchResult
# ---------------------------------------------------------------------------

class BatchResult:
    """Result from a batch processing operation.

    Attributes:
        request_id: Unique request identifier.
        success: Whether the operation succeeded.
        job_id: Batch job identifier.
        data: Batch result data payload.
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
        data: Dashboard data payload.
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
# BlockchainIntegrationService - Main facade
# ===========================================================================

class BlockchainIntegrationService:
    """Facade for the Blockchain Integration Agent (AGENT-EUDR-013).

    Provides a unified interface to all 8 engines:
        1. TransactionAnchor          - On-chain Merkle root anchoring
        2. SmartContractManager       - Contract deployment and lifecycle
        3. MultiChainConnector        - Multi-chain connection management
        4. VerificationEngine         - Anchor verification via proofs
        5. EventListener              - On-chain event indexing
        6. MerkleProofGenerator       - Merkle tree and proof generation
        7. CrossPartySharing          - Cross-party data access grants
        8. ComplianceEvidencePackager - Evidence package generation

    Singleton pattern with thread-safe initialization.

    Example:
        >>> service = BlockchainIntegrationService()
        >>> await service.startup()
        >>> result = await service.anchor_record({...})
        >>> await service.shutdown()
    """

    _instance: Optional[BlockchainIntegrationService] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize BlockchainIntegrationService.

        Loads configuration but does NOT start connections or engines.
        Call ``startup()`` to activate the service.
        """
        self._config: BlockchainIntegrationConfig = get_config()

        self._started = False
        self._start_time: Optional[float] = None
        self._config_hash = _compute_provenance_hash(
            self._config.database_url,
            self._config.redis_url,
            self._config.primary_chain,
            str(self._config.batch_size),
            self._config.genesis_hash,
        )

        # Connection handles
        self._db_pool: Optional[Any] = None
        self._redis: Optional[Any] = None

        # Engine instances (initialized in startup)
        self._transaction_anchor: Optional[Any] = None
        self._smart_contract_manager: Optional[Any] = None
        self._multi_chain_connector: Optional[Any] = None
        self._verification_engine: Optional[Any] = None
        self._event_listener: Optional[Any] = None
        self._merkle_proof_generator: Optional[Any] = None
        self._cross_party_sharing: Optional[Any] = None
        self._compliance_evidence_packager: Optional[Any] = None

        # Reference data (loaded in startup)
        self._ref_chain_configs: Optional[Dict[str, Any]] = None
        self._ref_contract_abis: Optional[List[str]] = None
        self._ref_anchor_rules: Optional[Dict[str, Any]] = None

        # Health check background task
        self._health_task: Optional[asyncio.Task[None]] = None
        self._last_health: Optional[HealthResult] = None

        # OpenTelemetry tracer
        self._tracer: Optional[Any] = None

        # Metrics counters
        self._metrics: Dict[str, int] = {
            "anchors_created": 0,
            "anchors_confirmed": 0,
            "anchors_failed": 0,
            "batches_anchored": 0,
            "verifications": 0,
            "verifications_tampered": 0,
            "merkle_trees_built": 0,
            "merkle_proofs_generated": 0,
            "contracts_deployed": 0,
            "contract_calls": 0,
            "events_indexed": 0,
            "access_grants": 0,
            "access_revocations": 0,
            "evidence_packages": 0,
            "batch_jobs": 0,
            "errors": 0,
        }

        logger.info(
            "BlockchainIntegrationService created: config_hash=%s, "
            "primary_chain=%s, batch_size=%d",
            self._config_hash[:12],
            self._config.primary_chain,
            self._config.batch_size,
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
    def config(self) -> BlockchainIntegrationConfig:
        """Return the service configuration."""
        return self._config

    @property
    def transaction_anchor(self) -> Any:
        """Return the TransactionAnchor engine instance."""
        self._ensure_started()
        return self._transaction_anchor

    @property
    def smart_contract_manager(self) -> Any:
        """Return the SmartContractManager engine instance."""
        self._ensure_started()
        return self._smart_contract_manager

    @property
    def multi_chain_connector(self) -> Any:
        """Return the MultiChainConnector engine instance."""
        self._ensure_started()
        return self._multi_chain_connector

    @property
    def verification_engine(self) -> Any:
        """Return the VerificationEngine instance."""
        self._ensure_started()
        return self._verification_engine

    @property
    def event_listener(self) -> Any:
        """Return the EventListener engine instance."""
        self._ensure_started()
        return self._event_listener

    @property
    def merkle_proof_generator(self) -> Any:
        """Return the MerkleProofGenerator engine instance."""
        self._ensure_started()
        return self._merkle_proof_generator

    @property
    def cross_party_sharing(self) -> Any:
        """Return the CrossPartySharing engine instance."""
        self._ensure_started()
        return self._cross_party_sharing

    @property
    def compliance_evidence_packager(self) -> Any:
        """Return the ComplianceEvidencePackager engine instance."""
        self._ensure_started()
        return self._compliance_evidence_packager

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
            logger.debug("BlockchainIntegrationService already started")
            return

        start = time.monotonic()
        logger.info("BlockchainIntegrationService starting up...")

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
            "BlockchainIntegrationService started in %.1fms: "
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
            logger.debug("BlockchainIntegrationService already stopped")
            return

        logger.info("BlockchainIntegrationService shutting down...")
        start = time.monotonic()

        self._stop_health_check()
        await self._close_engines()
        await self._close_redis()
        await self._close_database()
        self._flush_metrics()

        self._started = False
        elapsed = (time.monotonic() - start) * 1000
        logger.info(
            "BlockchainIntegrationService shut down in %.1fms", elapsed,
        )

    # ==================================================================
    # FACADE METHODS: Engine 1 - TransactionAnchor
    # ==================================================================

    async def anchor_record(
        self,
        anchor_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Anchor a single record on-chain via Merkle root submission.

        Delegates to TransactionAnchor.anchor().

        Args:
            anchor_data: Anchor record data including data_hash,
                event_type, operator_id, and chain.

        Returns:
            Dictionary with anchor creation result.
        """
        self._ensure_started()
        start = time.monotonic()
        try:
            result = self._safe_engine_call(
                self._transaction_anchor, "anchor", anchor_data,
            )
            self._metrics["anchors_created"] += 1
            chain = anchor_data.get("chain", self._config.primary_chain)
            event_type = anchor_data.get("event_type", "unknown")
            record_anchor_created(chain, event_type)
            elapsed = time.monotonic() - start
            observe_anchor_duration(elapsed)
            if result is None:
                return {"status": "engine_unavailable"}
            return self._wrap_result(result, start)
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("anchor")
            logger.error("anchor_record failed: %s", exc, exc_info=True)
            raise

    async def anchor_batch(
        self,
        records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Anchor multiple records as a batch Merkle root.

        Delegates to TransactionAnchor.anchor_batch().

        Args:
            records: List of anchor record data dictionaries.

        Returns:
            Dictionary with batch anchor results.
        """
        self._ensure_started()
        start = time.monotonic()
        try:
            result = self._safe_engine_call(
                self._transaction_anchor, "anchor_batch",
                {"records": records},
            )
            self._metrics["batches_anchored"] += 1
            self._metrics["anchors_created"] += len(records)
            if result is None:
                return {"status": "engine_unavailable"}
            return self._wrap_result(result, start)
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("anchor")
            logger.error("anchor_batch failed: %s", exc, exc_info=True)
            raise

    async def get_anchor(
        self,
        anchor_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get an anchor record by ID.

        Delegates to TransactionAnchor.get_anchor().

        Args:
            anchor_id: Anchor record identifier.

        Returns:
            Anchor record or None.
        """
        self._ensure_started()
        return self._safe_engine_call_with_args(
            self._transaction_anchor, "get_anchor",
            anchor_id=anchor_id,
        )

    async def get_anchor_status(
        self,
        anchor_id: str,
    ) -> Dict[str, Any]:
        """Get the current on-chain status of an anchor.

        Delegates to TransactionAnchor.get_anchor_status().

        Args:
            anchor_id: Anchor record identifier.

        Returns:
            Dictionary with anchor status (pending, submitted, confirmed, failed).
        """
        self._ensure_started()
        result = self._safe_engine_call_with_args(
            self._transaction_anchor, "get_anchor_status",
            anchor_id=anchor_id,
        )
        if result is None:
            return {"status": "engine_unavailable"}
        return result if isinstance(result, dict) else {"status": "unknown"}

    async def get_anchor_history(
        self,
        anchor_id: str,
    ) -> Dict[str, Any]:
        """Get the full status history of an anchor record.

        Delegates to TransactionAnchor.get_anchor_history().

        Args:
            anchor_id: Anchor record identifier.

        Returns:
            Dictionary with anchor status timeline.
        """
        self._ensure_started()
        result = self._safe_engine_call_with_args(
            self._transaction_anchor, "get_anchor_history",
            anchor_id=anchor_id,
        )
        if result is None:
            return {"status": "engine_unavailable", "history": []}
        return result if isinstance(result, dict) else {"history": []}

    # ==================================================================
    # FACADE METHODS: Engine 2 - SmartContractManager
    # ==================================================================

    async def deploy_contract(
        self,
        deploy_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Deploy a smart contract to the blockchain.

        Delegates to SmartContractManager.deploy().

        Args:
            deploy_data: Deployment data including contract_type,
                chain, and constructor arguments.

        Returns:
            Dictionary with deployment result.
        """
        self._ensure_started()
        start = time.monotonic()
        try:
            result = self._safe_engine_call(
                self._smart_contract_manager, "deploy", deploy_data,
            )
            self._metrics["contracts_deployed"] += 1
            chain = deploy_data.get("chain", self._config.primary_chain)
            contract_type = deploy_data.get("contract_type", "unknown")
            record_contract_deployed(chain, contract_type)
            if result is None:
                return {"status": "engine_unavailable"}
            return self._wrap_result(result, start)
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("deploy_contract")
            logger.error(
                "deploy_contract failed: %s", exc, exc_info=True,
            )
            raise

    async def get_contract(
        self,
        contract_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get a deployed contract by ID.

        Delegates to SmartContractManager.get_contract().

        Args:
            contract_id: Contract identifier.

        Returns:
            Contract record or None.
        """
        self._ensure_started()
        return self._safe_engine_call_with_args(
            self._smart_contract_manager, "get_contract",
            contract_id=contract_id,
        )

    async def call_contract(
        self,
        call_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Call a function on a deployed smart contract.

        Delegates to SmartContractManager.call_function().

        Args:
            call_data: Call data including contract_id, function_name,
                and arguments.

        Returns:
            Dictionary with call result.
        """
        self._ensure_started()
        start = time.monotonic()
        try:
            result = self._safe_engine_call(
                self._smart_contract_manager, "call_function", call_data,
            )
            self._metrics["contract_calls"] += 1
            if result is None:
                return {"status": "engine_unavailable"}
            return self._wrap_result(result, start)
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("call_contract")
            logger.error(
                "call_contract failed: %s", exc, exc_info=True,
            )
            raise

    async def get_contract_state(
        self,
        contract_id: str,
    ) -> Dict[str, Any]:
        """Get the current state of a deployed contract.

        Delegates to SmartContractManager.get_state().

        Args:
            contract_id: Contract identifier.

        Returns:
            Dictionary with contract state.
        """
        self._ensure_started()
        result = self._safe_engine_call_with_args(
            self._smart_contract_manager, "get_state",
            contract_id=contract_id,
        )
        if result is None:
            return {"status": "engine_unavailable"}
        return result if isinstance(result, dict) else {"state": "unknown"}

    async def list_contracts(self) -> Dict[str, Any]:
        """List all deployed contracts.

        Delegates to SmartContractManager.list_contracts().

        Returns:
            Dictionary with list of contracts.
        """
        self._ensure_started()
        result = self._safe_engine_call(
            self._smart_contract_manager, "list_contracts",
        )
        if result is None:
            return {"status": "engine_unavailable", "contracts": []}
        return result

    # ==================================================================
    # FACADE METHODS: Engine 3 - MultiChainConnector
    # ==================================================================

    async def connect_chain(
        self,
        chain_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Connect to a blockchain network.

        Delegates to MultiChainConnector.connect().

        Args:
            chain_data: Connection data including network, rpc_url,
                and authentication credentials.

        Returns:
            Dictionary with connection result.
        """
        self._ensure_started()
        start = time.monotonic()
        try:
            result = self._safe_engine_call(
                self._multi_chain_connector, "connect", chain_data,
            )
            if result is None:
                return {"status": "engine_unavailable"}
            return self._wrap_result(result, start)
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("connect_chain")
            logger.error(
                "connect_chain failed: %s", exc, exc_info=True,
            )
            raise

    async def get_chain_status(
        self,
        network: str,
    ) -> Dict[str, Any]:
        """Get the connection status of a blockchain network.

        Delegates to MultiChainConnector.get_status().

        Args:
            network: Network identifier.

        Returns:
            Dictionary with connection status.
        """
        self._ensure_started()
        result = self._safe_engine_call_with_args(
            self._multi_chain_connector, "get_status",
            network=network,
        )
        if result is None:
            config = get_chain_config(network)
            return {
                "status": "engine_unavailable",
                "network": network,
                "config_available": config is not None,
            }
        return result if isinstance(result, dict) else {"status": "unknown"}

    async def list_chains(self) -> Dict[str, Any]:
        """List all configured and connected chains.

        Delegates to MultiChainConnector.list_chains().

        Returns:
            Dictionary with list of chain connections.
        """
        self._ensure_started()
        result = self._safe_engine_call(
            self._multi_chain_connector, "list_chains",
        )
        if result is None:
            return {
                "status": "engine_unavailable",
                "supported_networks": SUPPORTED_NETWORKS,
            }
        return result

    async def estimate_gas(
        self,
        gas_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Estimate gas cost for a blockchain operation.

        Delegates to MultiChainConnector.estimate_gas().

        Args:
            gas_data: Gas estimation data including network,
                operation_type, and optional contract_type.

        Returns:
            Dictionary with gas estimate.
        """
        self._ensure_started()
        result = self._safe_engine_call(
            self._multi_chain_connector, "estimate_gas", gas_data,
        )
        if result is None:
            network = gas_data.get("network", self._config.primary_chain)
            operation = gas_data.get("operation_type", "anchor_single")
            estimates = GAS_ESTIMATES.get(network, {})
            return {
                "status": "engine_unavailable",
                "estimated_gas": estimates.get(operation, 0),
                "network": network,
                "operation": operation,
            }
        return result

    # ==================================================================
    # FACADE METHODS: Engine 4 - VerificationEngine
    # ==================================================================

    async def verify_record(
        self,
        verify_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Verify an anchor record against its on-chain root.

        Delegates to VerificationEngine.verify().

        Args:
            verify_data: Verification data including anchor_id or
                data_hash, and optional merkle_proof.

        Returns:
            Dictionary with verification result.
        """
        self._ensure_started()
        start = time.monotonic()
        try:
            result = self._safe_engine_call(
                self._verification_engine, "verify", verify_data,
            )
            self._metrics["verifications"] += 1
            status = "verified"
            if result and isinstance(result, dict):
                status = result.get("status", "verified")
            record_verification(status)
            elapsed = time.monotonic() - start
            observe_verification_duration(elapsed)
            if result is None:
                return {"status": "engine_unavailable"}
            return self._wrap_result(result, start)
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("verify")
            logger.error("verify_record failed: %s", exc, exc_info=True)
            raise

    async def verify_batch(
        self,
        records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Verify multiple anchor records in a batch.

        Delegates to VerificationEngine.batch_verify().

        Args:
            records: List of verification data dictionaries.

        Returns:
            Dictionary with batch verification results.
        """
        self._ensure_started()
        start = time.monotonic()
        try:
            result = self._safe_engine_call(
                self._verification_engine, "batch_verify",
                {"records": records},
            )
            self._metrics["verifications"] += len(records)
            if result is None:
                return {"status": "engine_unavailable"}
            return self._wrap_result(result, start)
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("verify")
            logger.error("verify_batch failed: %s", exc, exc_info=True)
            raise

    async def verify_merkle_proof(
        self,
        proof_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Verify a Merkle inclusion proof against an on-chain root.

        Delegates to VerificationEngine.verify_proof().

        Args:
            proof_data: Proof verification data including leaf_hash,
                proof_hashes, root_hash, and leaf_index.

        Returns:
            Dictionary with proof verification result.
        """
        self._ensure_started()
        start = time.monotonic()
        try:
            result = self._safe_engine_call(
                self._verification_engine, "verify_proof", proof_data,
            )
            self._metrics["verifications"] += 1
            elapsed = time.monotonic() - start
            observe_verification_duration(elapsed)
            if result is None:
                return {"status": "engine_unavailable"}
            return self._wrap_result(result, start)
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("verify")
            logger.error(
                "verify_merkle_proof failed: %s", exc, exc_info=True,
            )
            raise

    async def get_verification_result(
        self,
        verification_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get a stored verification result by ID.

        Delegates to VerificationEngine.get_result().

        Args:
            verification_id: Verification result identifier.

        Returns:
            Verification result or None.
        """
        self._ensure_started()
        return self._safe_engine_call_with_args(
            self._verification_engine, "get_result",
            verification_id=verification_id,
        )

    # ==================================================================
    # FACADE METHODS: Engine 5 - EventListener
    # ==================================================================

    async def subscribe_events(
        self,
        subscription_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Subscribe to on-chain events.

        Delegates to EventListener.subscribe().

        Args:
            subscription_data: Subscription data including chain,
                contract_address, and event_types.

        Returns:
            Dictionary with subscription result.
        """
        self._ensure_started()
        start = time.monotonic()
        try:
            result = self._safe_engine_call(
                self._event_listener, "subscribe", subscription_data,
            )
            if result is None:
                return {"status": "engine_unavailable"}
            return self._wrap_result(result, start)
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("subscribe_events")
            logger.error(
                "subscribe_events failed: %s", exc, exc_info=True,
            )
            raise

    async def unsubscribe_events(
        self,
        subscription_id: str,
    ) -> Dict[str, Any]:
        """Unsubscribe from on-chain events.

        Delegates to EventListener.unsubscribe().

        Args:
            subscription_id: Subscription identifier.

        Returns:
            Dictionary with unsubscribe result.
        """
        self._ensure_started()
        result = self._safe_engine_call_with_args(
            self._event_listener, "unsubscribe",
            subscription_id=subscription_id,
        )
        if result is None:
            return {"status": "engine_unavailable"}
        return result if isinstance(result, dict) else {"success": False}

    async def query_events(
        self,
        query_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Query indexed on-chain events.

        Delegates to EventListener.query().

        Args:
            query_data: Event query parameters including chain,
                event_type, from_block, and to_block.

        Returns:
            Dictionary with matching events.
        """
        self._ensure_started()
        start = time.monotonic()
        try:
            result = self._safe_engine_call(
                self._event_listener, "query", query_data,
            )
            if result is None:
                return {"status": "engine_unavailable", "events": []}
            return self._wrap_result(result, start)
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("query_events")
            logger.error("query_events failed: %s", exc, exc_info=True)
            raise

    async def get_event(
        self,
        event_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get a specific indexed event by ID.

        Delegates to EventListener.get_event().

        Args:
            event_id: Event identifier.

        Returns:
            Event record or None.
        """
        self._ensure_started()
        return self._safe_engine_call_with_args(
            self._event_listener, "get_event",
            event_id=event_id,
        )

    async def replay_events(
        self,
        replay_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Replay historical on-chain events from a block range.

        Delegates to EventListener.replay().

        Args:
            replay_data: Replay parameters including chain,
                from_block, to_block, and event_types.

        Returns:
            Dictionary with replayed events.
        """
        self._ensure_started()
        start = time.monotonic()
        try:
            result = self._safe_engine_call(
                self._event_listener, "replay", replay_data,
            )
            if result is None:
                return {"status": "engine_unavailable"}
            return self._wrap_result(result, start)
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("replay_events")
            logger.error("replay_events failed: %s", exc, exc_info=True)
            raise

    # ==================================================================
    # FACADE METHODS: Engine 6 - MerkleProofGenerator
    # ==================================================================

    async def build_merkle_tree(
        self,
        tree_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build a Merkle tree from a set of data hashes.

        Delegates to MerkleProofGenerator.build_tree().

        Args:
            tree_data: Tree construction data including leaf_hashes
                and optional hash_algorithm.

        Returns:
            Dictionary with Merkle tree including root hash.
        """
        self._ensure_started()
        start = time.monotonic()
        try:
            result = self._safe_engine_call(
                self._merkle_proof_generator, "build_tree", tree_data,
            )
            self._metrics["merkle_trees_built"] += 1
            record_merkle_tree_built()
            elapsed = time.monotonic() - start
            observe_merkle_build_duration(elapsed)
            if result is None:
                return {"status": "engine_unavailable"}
            return self._wrap_result(result, start)
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("build_tree")
            logger.error(
                "build_merkle_tree failed: %s", exc, exc_info=True,
            )
            raise

    async def get_merkle_tree(
        self,
        tree_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get a Merkle tree by ID.

        Delegates to MerkleProofGenerator.get_tree().

        Args:
            tree_id: Merkle tree identifier.

        Returns:
            Merkle tree record or None.
        """
        self._ensure_started()
        return self._safe_engine_call_with_args(
            self._merkle_proof_generator, "get_tree",
            tree_id=tree_id,
        )

    async def generate_proof(
        self,
        proof_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate a Merkle inclusion proof for a leaf.

        Delegates to MerkleProofGenerator.generate_proof().

        Args:
            proof_data: Proof generation data including tree_id
                and leaf_hash or leaf_index.

        Returns:
            Dictionary with Merkle proof data.
        """
        self._ensure_started()
        start = time.monotonic()
        try:
            result = self._safe_engine_call(
                self._merkle_proof_generator, "generate_proof",
                proof_data,
            )
            self._metrics["merkle_proofs_generated"] += 1
            record_merkle_proof_generated()
            if result is None:
                return {"status": "engine_unavailable"}
            return self._wrap_result(result, start)
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("generate_proof")
            logger.error(
                "generate_proof failed: %s", exc, exc_info=True,
            )
            raise

    async def verify_proof(
        self,
        proof_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Verify a Merkle inclusion proof locally (off-chain).

        Delegates to MerkleProofGenerator.verify_proof().

        Args:
            proof_data: Proof verification data including leaf_hash,
                proof_hashes, and root_hash.

        Returns:
            Dictionary with local proof verification result.
        """
        self._ensure_started()
        result = self._safe_engine_call(
            self._merkle_proof_generator, "verify_proof", proof_data,
        )
        if result is None:
            return {"status": "engine_unavailable", "valid": False}
        return result

    # ==================================================================
    # FACADE METHODS: Engine 7 - CrossPartySharing
    # ==================================================================

    async def grant_access(
        self,
        grant_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Grant a cross-party data access permission.

        Delegates to CrossPartySharing.grant_access().

        Args:
            grant_data: Grant data including grantee_id, access_level,
                anchor_ids, and optional expiry.

        Returns:
            Dictionary with grant creation result.
        """
        self._ensure_started()
        start = time.monotonic()
        try:
            result = self._safe_engine_call(
                self._cross_party_sharing, "grant_access", grant_data,
            )
            self._metrics["access_grants"] += 1
            access_level = grant_data.get("access_level", "auditor")
            record_access_grant(access_level)
            if result is None:
                return {"status": "engine_unavailable"}
            return self._wrap_result(result, start)
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("grant_access")
            logger.error(
                "grant_access failed: %s", exc, exc_info=True,
            )
            raise

    async def revoke_access(
        self,
        revoke_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Revoke a cross-party data access permission.

        Delegates to CrossPartySharing.revoke_access().

        Args:
            revoke_data: Revocation data including grant_id and
                optional reason.

        Returns:
            Dictionary with revocation result.
        """
        self._ensure_started()
        start = time.monotonic()
        try:
            result = self._safe_engine_call(
                self._cross_party_sharing, "revoke_access", revoke_data,
            )
            self._metrics["access_revocations"] += 1
            if result is None:
                return {"status": "engine_unavailable"}
            return self._wrap_result(result, start)
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("revoke_access")
            logger.error(
                "revoke_access failed: %s", exc, exc_info=True,
            )
            raise

    async def list_grants(
        self,
        filter_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """List access grants with optional filters.

        Delegates to CrossPartySharing.list_grants().

        Args:
            filter_data: Optional filter parameters including
                grantee_id, access_level, and status.

        Returns:
            Dictionary with list of grants.
        """
        self._ensure_started()
        result = self._safe_engine_call(
            self._cross_party_sharing, "list_grants",
            filter_data or {},
        )
        if result is None:
            return {"status": "engine_unavailable", "grants": []}
        return result

    async def request_access(
        self,
        request_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Request cross-party data access from a data owner.

        Delegates to CrossPartySharing.request_access().

        Args:
            request_data: Access request data including requester_id,
                access_level, and justification.

        Returns:
            Dictionary with access request result.
        """
        self._ensure_started()
        start = time.monotonic()
        try:
            result = self._safe_engine_call(
                self._cross_party_sharing, "request_access",
                request_data,
            )
            if result is None:
                return {"status": "engine_unavailable"}
            return self._wrap_result(result, start)
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("request_access")
            logger.error(
                "request_access failed: %s", exc, exc_info=True,
            )
            raise

    async def confirm_action(
        self,
        confirmation_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Confirm a multi-party action (e.g. grant approval).

        Delegates to CrossPartySharing.confirm_action().

        Args:
            confirmation_data: Confirmation data including action_id,
                confirmer_id, and action_type.

        Returns:
            Dictionary with confirmation result.
        """
        self._ensure_started()
        result = self._safe_engine_call(
            self._cross_party_sharing, "confirm_action",
            confirmation_data,
        )
        if result is None:
            return {"status": "engine_unavailable"}
        return result

    # ==================================================================
    # FACADE METHODS: Engine 8 - ComplianceEvidencePackager
    # ==================================================================

    async def generate_evidence_package(
        self,
        package_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate a compliance evidence package for Article 14.

        Delegates to ComplianceEvidencePackager.generate_package().

        Args:
            package_data: Package data including dds_id, anchor_ids,
                format, and optional sections.

        Returns:
            Dictionary with evidence package result.
        """
        self._ensure_started()
        start = time.monotonic()
        try:
            result = self._safe_engine_call(
                self._compliance_evidence_packager,
                "generate_package", package_data,
            )
            self._metrics["evidence_packages"] += 1
            fmt = package_data.get("format", "json")
            record_evidence_package(fmt)
            if result is None:
                return {"status": "engine_unavailable"}
            return self._wrap_result(result, start)
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("create_evidence")
            logger.error(
                "generate_evidence_package failed: %s", exc,
                exc_info=True,
            )
            raise

    async def get_evidence_package(
        self,
        package_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get an evidence package by ID.

        Delegates to ComplianceEvidencePackager.get_package().

        Args:
            package_id: Evidence package identifier.

        Returns:
            Evidence package record or None.
        """
        self._ensure_started()
        return self._safe_engine_call_with_args(
            self._compliance_evidence_packager, "get_package",
            package_id=package_id,
        )

    async def download_evidence(
        self,
        package_id: str,
    ) -> Dict[str, Any]:
        """Download an evidence package with content.

        Delegates to ComplianceEvidencePackager.download().

        Args:
            package_id: Evidence package identifier.

        Returns:
            Dictionary with evidence package content.
        """
        self._ensure_started()
        result = self._safe_engine_call_with_args(
            self._compliance_evidence_packager, "download",
            package_id=package_id,
        )
        if result is None:
            return {"status": "engine_unavailable"}
        return result if isinstance(result, dict) else {"data": result}

    async def verify_evidence(
        self,
        evidence_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Verify an evidence package signature and integrity.

        Delegates to ComplianceEvidencePackager.verify_package().

        Args:
            evidence_data: Evidence verification data including
                package_id or package_content.

        Returns:
            Dictionary with evidence verification result.
        """
        self._ensure_started()
        start = time.monotonic()
        try:
            result = self._safe_engine_call(
                self._compliance_evidence_packager,
                "verify_package", evidence_data,
            )
            if result is None:
                return {"status": "engine_unavailable"}
            return self._wrap_result(result, start)
        except Exception as exc:
            self._metrics["errors"] += 1
            record_api_error("verify_evidence")
            logger.error(
                "verify_evidence failed: %s", exc, exc_info=True,
            )
            raise

    # ==================================================================
    # FACADE METHODS: Batch operations
    # ==================================================================

    async def submit_batch_job(
        self,
        batch_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Submit a batch processing job.

        Supports batch anchoring, batch verification, and batch
        evidence generation across multiple records.

        Args:
            batch_data: Batch job data including job_type, records,
                and optional concurrency settings.

        Returns:
            Dictionary with batch job submission result.
        """
        self._ensure_started()
        start = time.monotonic()
        job_id = f"BATCH-{uuid.uuid4().hex[:8]}"

        max_size = self._config.batch_max_size
        records = batch_data.get("records", [])
        if len(records) > max_size:
            return {
                "success": False,
                "job_id": job_id,
                "error": (
                    f"Batch size {len(records)} exceeds maximum "
                    f"{max_size}"
                ),
            }

        try:
            job_type = batch_data.get("job_type", "anchor")

            if job_type == "anchor":
                result = await self.anchor_batch(records)
            elif job_type == "verify":
                result = await self.verify_batch(records)
            else:
                result = {"error": f"Unknown job type: {job_type}"}

            self._metrics["batch_jobs"] += 1
            elapsed_ms = (time.monotonic() - start) * 1000

            return {
                "success": True,
                "job_id": job_id,
                "job_type": job_type,
                "total_records": len(records),
                "result": result,
                "processing_time_ms": round(elapsed_ms, 2),
                "provenance_hash": _compute_provenance_hash(
                    job_id, str(len(records)), job_type,
                ),
            }

        except Exception as exc:
            self._metrics["errors"] += 1
            logger.error(
                "submit_batch_job failed: %s", exc, exc_info=True,
            )
            raise

    async def cancel_batch_job(
        self,
        job_id: str,
    ) -> Dict[str, Any]:
        """Cancel a running batch job.

        Args:
            job_id: Batch job identifier.

        Returns:
            Dictionary with cancellation result.
        """
        self._ensure_started()
        # Batch cancellation is a best-effort operation
        logger.info("Batch job cancellation requested: job_id=%s", job_id)
        return {
            "success": True,
            "job_id": job_id,
            "status": "cancellation_requested",
        }

    # ==================================================================
    # Statistics and health
    # ==================================================================

    async def get_dashboard(self) -> Dict[str, Any]:
        """Get an overview dashboard of blockchain integration activity.

        Aggregates metrics from all engines into a single dashboard
        view suitable for the EUDR compliance dashboard.

        Returns:
            Dictionary with dashboard data.
        """
        self._ensure_started()
        return {
            "agent_id": _AGENT_ID,
            "version": _MODULE_VERSION,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "timestamp": utcnow().isoformat(),
            "metrics": dict(self._metrics),
            "engines_active": self._count_initialized_engines(),
            "engines_total": _ENGINE_COUNT,
            "config_hash": self._config_hash[:12],
            "primary_chain": self._config.primary_chain,
            "fallback_chain": self._config.fallback_chain,
        }

    async def get_stats(self) -> Dict[str, Any]:
        """Get service statistics and counters.

        Returns:
            Dictionary with service metrics and counters.
        """
        return {
            "agent_id": _AGENT_ID,
            "version": _MODULE_VERSION,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "timestamp": utcnow().isoformat(),
            "metrics": dict(self._metrics),
            "engines_active": self._count_initialized_engines(),
            "engines_total": _ENGINE_COUNT,
            "reference_data": {
                "chain_configs": (
                    len(self._ref_chain_configs)
                    if self._ref_chain_configs else 0
                ),
                "contract_types": (
                    len(self._ref_contract_abis)
                    if self._ref_contract_abis else 0
                ),
                "anchor_rules": (
                    len(self._ref_anchor_rules)
                    if self._ref_anchor_rules else 0
                ),
            },
        }

    async def get_health(self) -> Dict[str, Any]:
        """Alias for health_check(). Returns comprehensive health status."""
        return await self.health_check()

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

        health = HealthResult(
            status=overall,
            checks=checks,
            timestamp=utcnow(),
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
            "greenlang.agents.eudr.blockchain_integration"
        ).setLevel(log_level)
        logger.debug("Logging configured: level=%s", self._config.log_level)

    def _init_tracer(self) -> None:
        """Initialize OpenTelemetry tracer if available."""
        if OTEL_AVAILABLE and otel_trace is not None:
            try:
                self._tracer = otel_trace.get_tracer(
                    "greenlang.agents.eudr.blockchain_integration",
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
            self._ref_chain_configs = CHAIN_CONFIGS
            logger.debug(
                "Loaded chain configs: %d networks",
                len(self._ref_chain_configs)
                if self._ref_chain_configs else 0,
            )
        except Exception as exc:
            logger.warning("Failed to load chain configs: %s", exc)

        try:
            self._ref_contract_abis = CONTRACT_TYPES
            logger.debug(
                "Loaded contract ABIs: %d types",
                len(self._ref_contract_abis)
                if self._ref_contract_abis else 0,
            )
        except Exception as exc:
            logger.warning("Failed to load contract ABIs: %s", exc)

        try:
            self._ref_anchor_rules = ANCHOR_RULES
            logger.debug(
                "Loaded anchor rules: %d event types",
                len(self._ref_anchor_rules)
                if self._ref_anchor_rules else 0,
            )
        except Exception as exc:
            logger.warning("Failed to load anchor rules: %s", exc)

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

        # Engine 1: TransactionAnchor
        try:
            from greenlang.agents.eudr.blockchain_integration.transaction_anchor import (
                TransactionAnchor,
            )
            self._transaction_anchor = TransactionAnchor(config=config)
            logger.debug("Engine 1 initialized: TransactionAnchor")
        except (ImportError, Exception) as exc:
            logger.warning(
                "Engine 1 (TransactionAnchor) init failed: %s", exc,
            )

        # Engine 2: SmartContractManager
        try:
            from greenlang.agents.eudr.blockchain_integration.smart_contract_manager import (
                SmartContractManager,
            )
            self._smart_contract_manager = SmartContractManager(config=config)
            logger.debug("Engine 2 initialized: SmartContractManager")
        except (ImportError, Exception) as exc:
            logger.warning(
                "Engine 2 (SmartContractManager) init failed: %s", exc,
            )

        # Engine 3: MultiChainConnector
        try:
            from greenlang.agents.eudr.blockchain_integration.multi_chain_connector import (
                MultiChainConnector,
            )
            self._multi_chain_connector = MultiChainConnector(config=config)
            logger.debug("Engine 3 initialized: MultiChainConnector")
        except (ImportError, Exception) as exc:
            logger.warning(
                "Engine 3 (MultiChainConnector) init failed: %s", exc,
            )

        # Engine 4: VerificationEngine
        try:
            from greenlang.agents.eudr.blockchain_integration.verification_engine import (
                VerificationEngine,
            )
            self._verification_engine = VerificationEngine(config=config)
            logger.debug("Engine 4 initialized: VerificationEngine")
        except (ImportError, Exception) as exc:
            logger.warning(
                "Engine 4 (VerificationEngine) init failed: %s", exc,
            )

        # Engine 5: EventListener
        try:
            from greenlang.agents.eudr.blockchain_integration.event_listener import (
                EventListener,
            )
            self._event_listener = EventListener(config=config)
            logger.debug("Engine 5 initialized: EventListener")
        except (ImportError, Exception) as exc:
            logger.warning(
                "Engine 5 (EventListener) init failed: %s", exc,
            )

        # Engine 6: MerkleProofGenerator
        try:
            from greenlang.agents.eudr.blockchain_integration.merkle_proof_generator import (
                MerkleProofGenerator,
            )
            self._merkle_proof_generator = MerkleProofGenerator(config=config)
            logger.debug("Engine 6 initialized: MerkleProofGenerator")
        except (ImportError, Exception) as exc:
            logger.warning(
                "Engine 6 (MerkleProofGenerator) init failed: %s", exc,
            )

        # Engine 7: CrossPartySharing
        try:
            from greenlang.agents.eudr.blockchain_integration.cross_party_sharing import (
                CrossPartySharing,
            )
            self._cross_party_sharing = CrossPartySharing(config=config)
            logger.debug("Engine 7 initialized: CrossPartySharing")
        except (ImportError, Exception) as exc:
            logger.warning(
                "Engine 7 (CrossPartySharing) init failed: %s", exc,
            )

        # Engine 8: ComplianceEvidencePackager
        try:
            from greenlang.agents.eudr.blockchain_integration.compliance_evidence_packager import (
                ComplianceEvidencePackager,
            )
            self._compliance_evidence_packager = ComplianceEvidencePackager(
                config=config,
            )
            logger.debug("Engine 8 initialized: ComplianceEvidencePackager")
        except (ImportError, Exception) as exc:
            logger.warning(
                "Engine 8 (ComplianceEvidencePackager) init failed: %s", exc,
            )

        count = self._count_initialized_engines()
        logger.info("Engines initialized: %d/%d", count, _ENGINE_COUNT)

    async def _close_engines(self) -> None:
        """Close all engines and release resources."""
        engine_names = [
            "_transaction_anchor",
            "_smart_contract_manager",
            "_multi_chain_connector",
            "_verification_engine",
            "_event_listener",
            "_merkle_proof_generator",
            "_cross_party_sharing",
            "_compliance_evidence_packager",
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
            "transaction_anchor": self._transaction_anchor,
            "smart_contract_manager": self._smart_contract_manager,
            "multi_chain_connector": self._multi_chain_connector,
            "verification_engine": self._verification_engine,
            "event_listener": self._event_listener,
            "merkle_proof_generator": self._merkle_proof_generator,
            "cross_party_sharing": self._cross_party_sharing,
            "compliance_evidence_packager": self._compliance_evidence_packager,
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
            self._ref_chain_configs,
            self._ref_contract_abis,
            self._ref_anchor_rules,
        ] if x is not None)
        return {
            "status": "healthy" if loaded == 3 else "degraded",
            "loaded_datasets": loaded,
            "total_datasets": 3,
        }

    def _count_initialized_engines(self) -> int:
        """Count the number of successfully initialized engines."""
        engines = [
            self._transaction_anchor,
            self._smart_contract_manager,
            self._multi_chain_connector,
            self._verification_engine,
            self._event_listener,
            self._merkle_proof_generator,
            self._cross_party_sharing,
            self._compliance_evidence_packager,
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
                "BlockchainIntegrationService is not started. "
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
    """FastAPI lifespan context manager for the Blockchain Integration service.

    Automatically starts the service on application startup and shuts it
    down on application shutdown.  The service instance is stored in
    ``app.state.bci_service`` for access from route handlers.

    Usage with FastAPI::

        from fastapi import FastAPI
        from greenlang.agents.eudr.blockchain_integration.setup import lifespan
from greenlang.schemas import utcnow

        app = FastAPI(lifespan=lifespan)

    Args:
        app: The FastAPI application instance.

    Yields:
        None (service is accessible via ``app.state.bci_service``).
    """
    service = get_service()
    app.state.bci_service = service
    try:
        await service.startup()
        yield
    finally:
        await service.shutdown()

# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_service_instance: Optional[BlockchainIntegrationService] = None
_service_lock = threading.Lock()

def get_service() -> BlockchainIntegrationService:
    """Return the singleton BlockchainIntegrationService instance.

    Uses double-checked locking for thread safety.  The instance is
    created on first call.

    Returns:
        BlockchainIntegrationService singleton instance.

    Example:
        >>> service = get_service()
        >>> await service.startup()
    """
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = BlockchainIntegrationService()
    return _service_instance

def set_service(service: BlockchainIntegrationService) -> None:
    """Replace the singleton BlockchainIntegrationService instance.

    Primarily intended for testing and dependency injection.

    Args:
        service: Replacement service instance.
    """
    global _service_instance
    with _service_lock:
        _service_instance = service
    logger.info("BlockchainIntegrationService singleton replaced")

def reset_service() -> None:
    """Reset the singleton BlockchainIntegrationService to None.

    The next call to ``get_service()`` will create a fresh instance.
    Intended for test teardown.
    """
    global _service_instance
    with _service_lock:
        _service_instance = None
    logger.debug("BlockchainIntegrationService singleton reset")

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Service
    "BlockchainIntegrationService",
    "HealthResult",
    "lifespan",
    "get_service",
    "set_service",
    "reset_service",
    # Result containers
    "AnchorResult",
    "ContractResult",
    "ChainResult",
    "VerifyResult",
    "EventResult",
    "MerkleResult",
    "SharingResult",
    "EvidenceResult",
    "BatchResult",
    "DashboardResult",
]
