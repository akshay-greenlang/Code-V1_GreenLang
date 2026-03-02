# -*- coding: utf-8 -*-
"""
AuditEventEngine - Immutable Audit Event Recording with SHA-256 Hash Chains

Engine 1 of 7 for AGENT-MRV-030 (GL-MRV-X-042).

Records tamper-evident audit events for all MRV emissions calculations.
Every event is chained via SHA-256 hashing to create an immutable log.
One chain exists per organization per reporting year, anchored to a
deterministic genesis hash.

Features:
    - Genesis-anchored hash chains (one per organization per reporting year)
    - 12 audit event types covering full MRV lifecycle
    - Event payload validation with schema enforcement
    - Chain integrity verification (forward and backward)
    - Concurrent-safe event appending with RLock
    - Event querying by type, time range, agent, scope, category
    - Chain export for external audit systems
    - Batch event recording for historical data

Hash Chain Algorithm:
    hash_input = f"{event_id}|{prev_hash}|{event_type}|{timestamp_iso}|{canonical_json(payload)}"
    event_hash = SHA-256(hash_input.encode('utf-8')).hexdigest()

Zero-Hallucination Guarantee:
    - No LLM involvement in event recording or hashing
    - Deterministic SHA-256 with canonical JSON (sorted keys)
    - All timestamps in UTC ISO format
    - Every hash is reproducible given the same inputs

Thread Safety:
    Uses __new__ singleton pattern with threading.Lock for thread-safe
    instantiation.  All chain mutations are protected by a dedicated
    threading.RLock.

Example:
    >>> from greenlang.audit_trail_lineage.audit_event_engine import (
    ...     AuditEventEngine,
    ... )
    >>> engine = AuditEventEngine()
    >>> result = engine.record_event(
    ...     event_type="CALCULATION_COMPLETED",
    ...     agent_id="GL-MRV-S1-001",
    ...     scope="scope_1",
    ...     category=None,
    ...     organization_id="org-001",
    ...     reporting_year=2025,
    ...     calculation_id="calc-abc",
    ...     payload={"total_co2e": "1234.56"},
    ...     data_quality_score=Decimal("0.85"),
    ... )
    >>> print(result["event_hash"])

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-X-042
"""

import hashlib
import json
import logging
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

AGENT_ID: str = "GL-MRV-X-042"
AGENT_COMPONENT: str = "AGENT-MRV-030"
ENGINE_ID: str = "gl_atl_audit_event_engine"
ENGINE_VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_atl_"

GENESIS_HASH: str = "greenlang-atl-genesis-v1"
HASH_ALGORITHM: str = "sha256"
ENCODING: str = "utf-8"

# Decimal quantization constants
_ZERO = Decimal("0")
_ONE = Decimal("1")
_QUANT_2DP = Decimal("0.01")
_QUANT_4DP = Decimal("0.0001")

# Default limits
_DEFAULT_QUERY_LIMIT: int = 1000
_DEFAULT_QUERY_OFFSET: int = 0
_MAX_QUERY_LIMIT: int = 10000
_MAX_BATCH_SIZE: int = 5000


# =============================================================================
# ENUMERATIONS
# =============================================================================


class AuditEventType(str, Enum):
    """Audit event types covering the full MRV lifecycle.

    Twelve event types track every significant action from data intake
    through final report generation.
    """

    DATA_INGESTED = "DATA_INGESTED"
    DATA_VALIDATED = "DATA_VALIDATED"
    DATA_TRANSFORMED = "DATA_TRANSFORMED"
    EMISSION_FACTOR_RESOLVED = "EMISSION_FACTOR_RESOLVED"
    CALCULATION_STARTED = "CALCULATION_STARTED"
    CALCULATION_COMPLETED = "CALCULATION_COMPLETED"
    CALCULATION_FAILED = "CALCULATION_FAILED"
    COMPLIANCE_CHECKED = "COMPLIANCE_CHECKED"
    REPORT_GENERATED = "REPORT_GENERATED"
    PROVENANCE_SEALED = "PROVENANCE_SEALED"
    MANUAL_OVERRIDE = "MANUAL_OVERRIDE"
    CHAIN_VERIFIED = "CHAIN_VERIFIED"


# Pre-compute the set for O(1) membership checks
_VALID_EVENT_TYPES: Set[str] = {e.value for e in AuditEventType}

# Valid scopes
_VALID_SCOPES: Set[str] = {
    "scope_1",
    "scope_2",
    "scope_3",
}

# Valid Scope 3 category numbers (1-15)
_VALID_CATEGORIES: Set[int] = set(range(1, 16))


# =============================================================================
# DATA MODELS
# =============================================================================


@dataclass(frozen=True)
class AuditEventRecord:
    """Immutable record of a single audit event in the hash chain.

    Frozen to guarantee that once created, no field can be mutated.
    This ensures tamper evidence for the audit trail.

    Attributes:
        event_id: UUID-based unique identifier for this event.
        event_type: One of the 12 defined AuditEventType values.
        agent_id: Identifier of the agent that produced the event.
        scope: GHG scope (scope_1, scope_2, scope_3) or None.
        category: Scope 3 category number (1-15) or None.
        organization_id: Organization that owns this event.
        reporting_year: Reporting year for this event.
        calculation_id: Optional calculation identifier linking events.
        data_quality_score: Data quality score (0.00 to 1.00).
        payload: Event-specific payload data.
        prev_event_hash: SHA-256 hash of the previous event in chain.
        event_hash: SHA-256 hash of this event.
        chain_position: Zero-based position in the chain.
        timestamp: ISO 8601 UTC timestamp.
        metadata: Additional context (tags, notes, etc.).
    """

    event_id: str
    event_type: str
    agent_id: str
    scope: Optional[str]
    category: Optional[int]
    organization_id: str
    reporting_year: int
    calculation_id: Optional[str]
    data_quality_score: Decimal
    payload: Dict[str, Any]
    prev_event_hash: str
    event_hash: str
    chain_position: int
    timestamp: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to a plain dictionary.

        Returns:
            Dictionary representation with Decimal serialized as string.
        """
        d = asdict(self)
        d["data_quality_score"] = str(self.data_quality_score)
        return d


# =============================================================================
# SERIALIZATION HELPERS
# =============================================================================


def _decimal_serializer(obj: Any) -> Any:
    """JSON serialization hook for Decimal, datetime, and Enum.

    Args:
        obj: Object that the default JSON encoder cannot handle.

    Returns:
        Serializable representation.

    Raises:
        TypeError: If the object type is not supported.
    """
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# =============================================================================
# ENGINE
# =============================================================================


class AuditEventEngine:
    """Immutable audit event engine with SHA-256 hash chains.

    Implements a thread-safe singleton that records tamper-evident audit
    events for all MRV emissions calculations.  Events are organized into
    chains keyed by ``{organization_id}:{reporting_year}``, with each chain
    anchored to a deterministic genesis hash.

    The engine is designed for in-memory operation in development and
    testing.  In production, a database-backed adapter replaces the
    in-memory stores while preserving the identical hashing semantics.

    Thread Safety:
        Singleton instantiation is protected by ``threading.Lock``.
        All chain mutations are serialized through ``threading.RLock``
        (re-entrant to support nested calls like batch recording).

    Example:
        >>> engine = AuditEventEngine()
        >>> result = engine.record_event(
        ...     event_type="DATA_INGESTED",
        ...     agent_id="GL-MRV-S1-001",
        ...     scope="scope_1",
        ...     category=None,
        ...     organization_id="org-demo",
        ...     reporting_year=2025,
        ...     calculation_id=None,
        ...     payload={"rows": 500, "source": "ERP"},
        ...     data_quality_score=Decimal("0.90"),
        ... )
        >>> assert result["success"] is True
    """

    # Singleton machinery
    _instance: Optional["AuditEventEngine"] = None
    _singleton_lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "AuditEventEngine":
        """Thread-safe singleton instantiation."""
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        """Initialize engine state (runs only once due to singleton guard)."""
        if self._initialized:
            return
        self._initialized: bool = True

        # Chain storage: key = "{org_id}:{year}"
        self._chains: Dict[str, List[AuditEventRecord]] = {}
        self._chain_heads: Dict[str, str] = {}
        self._chain_positions: Dict[str, int] = {}

        # Event index: key = event_id
        self._event_index: Dict[str, AuditEventRecord] = {}

        # Calculation index: key = calculation_id -> list of event_ids
        self._calculation_index: Dict[str, List[str]] = defaultdict(list)

        # Re-entrant lock for all mutations
        self._lock: threading.RLock = threading.RLock()

        logger.info(
            "AuditEventEngine initialized | engine=%s | version=%s | agent=%s",
            ENGINE_ID,
            ENGINE_VERSION,
            AGENT_ID,
        )

    # =========================================================================
    # PUBLIC API -- Event Recording
    # =========================================================================

    def record_event(
        self,
        event_type: str,
        agent_id: str,
        scope: Optional[str],
        category: Optional[int],
        organization_id: str,
        reporting_year: int,
        calculation_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        data_quality_score: Optional[Decimal] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record a single immutable audit event.

        Creates a new event, computes its SHA-256 hash chained to the
        previous event in the same organization/year chain, and appends
        it atomically.

        Args:
            event_type: One of the 12 AuditEventType values.
            agent_id: Identifier of the originating agent.
            scope: GHG scope (scope_1, scope_2, scope_3) or None.
            category: Scope 3 category number (1-15) or None.
            organization_id: Organization identifier.
            reporting_year: Reporting year (e.g. 2025).
            calculation_id: Optional calculation identifier.
            payload: Event-specific data dictionary.
            data_quality_score: Quality score 0.00-1.00 (default 0.00).
            metadata: Optional additional context.

        Returns:
            Dictionary with keys: success, event_id, event_hash,
            chain_position, chain_key, timestamp, processing_time_ms.

        Raises:
            ValueError: If any input fails validation.
        """
        start_ns = time.monotonic_ns()

        # -- Validate inputs --------------------------------------------------
        self._validate_event_type(event_type)
        self._validate_organization_id(organization_id)
        self._validate_reporting_year(reporting_year)
        self._validate_scope(scope)
        self._validate_category(category)
        self._validate_agent_id(agent_id)

        safe_payload = payload if payload is not None else {}
        safe_metadata = metadata if metadata is not None else {}
        dq_score = self._normalize_dq_score(data_quality_score)

        # -- Build event under lock -------------------------------------------
        event_id = self._generate_event_id()
        timestamp = datetime.now(timezone.utc).isoformat()
        chain_key = self._get_chain_key(organization_id, reporting_year)

        with self._lock:
            prev_hash = self._chain_heads.get(chain_key, GENESIS_HASH)
            position = self._chain_positions.get(chain_key, 0)

            event_hash = self._compute_event_hash(
                event_id, prev_hash, event_type, timestamp, safe_payload,
            )

            record = AuditEventRecord(
                event_id=event_id,
                event_type=event_type,
                agent_id=agent_id,
                scope=scope,
                category=category,
                organization_id=organization_id,
                reporting_year=reporting_year,
                calculation_id=calculation_id,
                data_quality_score=dq_score,
                payload=safe_payload,
                prev_event_hash=prev_hash,
                event_hash=event_hash,
                chain_position=position,
                timestamp=timestamp,
                metadata=safe_metadata,
            )

            # Append to chain
            if chain_key not in self._chains:
                self._chains[chain_key] = []
            self._chains[chain_key].append(record)
            self._chain_heads[chain_key] = event_hash
            self._chain_positions[chain_key] = position + 1

            # Update indexes
            self._event_index[event_id] = record
            if calculation_id:
                self._calculation_index[calculation_id].append(event_id)

        elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000

        logger.info(
            "Audit event recorded | event_id=%s | type=%s | agent=%s | "
            "chain=%s | pos=%d | elapsed=%.3fms",
            event_id,
            event_type,
            agent_id,
            chain_key,
            position,
            elapsed_ms,
        )

        return {
            "success": True,
            "event_id": event_id,
            "event_type": event_type,
            "event_hash": event_hash,
            "prev_event_hash": prev_hash,
            "chain_position": position,
            "chain_key": chain_key,
            "timestamp": timestamp,
            "processing_time_ms": round(elapsed_ms, 3),
        }

    def record_batch(
        self,
        events: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Record a batch of audit events atomically.

        All events in the batch are appended to their respective chains
        within a single lock acquisition, ensuring atomicity.  If any
        single event fails validation the entire batch is rejected.

        Args:
            events: List of event dictionaries.  Each dictionary must
                contain the same keys accepted by ``record_event``.

        Returns:
            Dictionary with keys: success, total_recorded, event_ids,
            errors, processing_time_ms.

        Raises:
            ValueError: If the batch is empty, exceeds the maximum size,
                or any event fails validation.
        """
        start_ns = time.monotonic_ns()

        if not events:
            raise ValueError("Batch must contain at least one event")
        if len(events) > _MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(events)} exceeds maximum of {_MAX_BATCH_SIZE}"
            )

        # Pre-validate every event before acquiring the lock
        validated: List[Dict[str, Any]] = []
        for idx, evt in enumerate(events):
            try:
                self._validate_batch_event_dict(evt, idx)
                validated.append(evt)
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(
                    f"Batch event at index {idx} failed validation: {exc}"
                ) from exc

        # Record all events atomically
        recorded_ids: List[str] = []
        recorded_results: List[Dict[str, Any]] = []

        with self._lock:
            for evt in validated:
                result = self._record_event_internal(
                    event_type=evt["event_type"],
                    agent_id=evt["agent_id"],
                    scope=evt.get("scope"),
                    category=evt.get("category"),
                    organization_id=evt["organization_id"],
                    reporting_year=evt["reporting_year"],
                    calculation_id=evt.get("calculation_id"),
                    payload=evt.get("payload", {}),
                    data_quality_score=evt.get("data_quality_score"),
                    metadata=evt.get("metadata", {}),
                )
                recorded_ids.append(result["event_id"])
                recorded_results.append(result)

        elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000

        logger.info(
            "Batch recorded | count=%d | elapsed=%.3fms",
            len(recorded_ids),
            elapsed_ms,
        )

        return {
            "success": True,
            "total_recorded": len(recorded_ids),
            "event_ids": recorded_ids,
            "events": recorded_results,
            "processing_time_ms": round(elapsed_ms, 3),
        }

    # =========================================================================
    # PUBLIC API -- Event Retrieval
    # =========================================================================

    def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single audit event by its identifier.

        Args:
            event_id: UUID-based event identifier.

        Returns:
            Event dictionary if found, None otherwise.
        """
        if not event_id or not isinstance(event_id, str):
            return None

        with self._lock:
            record = self._event_index.get(event_id)

        if record is None:
            return None
        return record.to_dict()

    def get_events(
        self,
        organization_id: str,
        reporting_year: int,
        event_type: Optional[str] = None,
        agent_id: Optional[str] = None,
        scope: Optional[str] = None,
        category: Optional[int] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = _DEFAULT_QUERY_LIMIT,
        offset: int = _DEFAULT_QUERY_OFFSET,
    ) -> Dict[str, Any]:
        """Query audit events with optional filters.

        Events are returned in chain order (ascending chain_position).

        Args:
            organization_id: Organization identifier (required).
            reporting_year: Reporting year (required).
            event_type: Filter by event type.
            agent_id: Filter by originating agent.
            scope: Filter by GHG scope.
            category: Filter by Scope 3 category number.
            start_time: ISO 8601 lower bound (inclusive).
            end_time: ISO 8601 upper bound (inclusive).
            limit: Maximum events to return (default 1000, max 10000).
            offset: Number of events to skip (default 0).

        Returns:
            Dictionary with keys: success, events, total_matching,
            returned_count, has_more, filters_applied.
        """
        self._validate_organization_id(organization_id)
        self._validate_reporting_year(reporting_year)

        limit = min(max(1, limit), _MAX_QUERY_LIMIT)
        offset = max(0, offset)

        chain_key = self._get_chain_key(organization_id, reporting_year)

        with self._lock:
            chain = self._chains.get(chain_key, [])

        # Apply filters
        filtered = self._apply_filters(
            chain,
            event_type=event_type,
            agent_id=agent_id,
            scope=scope,
            category=category,
            start_time=start_time,
            end_time=end_time,
        )

        total_matching = len(filtered)
        page = filtered[offset: offset + limit]

        filters_applied = {}
        if event_type is not None:
            filters_applied["event_type"] = event_type
        if agent_id is not None:
            filters_applied["agent_id"] = agent_id
        if scope is not None:
            filters_applied["scope"] = scope
        if category is not None:
            filters_applied["category"] = category
        if start_time is not None:
            filters_applied["start_time"] = start_time
        if end_time is not None:
            filters_applied["end_time"] = end_time

        return {
            "success": True,
            "events": [r.to_dict() for r in page],
            "total_matching": total_matching,
            "returned_count": len(page),
            "has_more": (offset + limit) < total_matching,
            "limit": limit,
            "offset": offset,
            "chain_key": chain_key,
            "filters_applied": filters_applied,
        }

    def get_events_by_calculation(
        self, calculation_id: str,
    ) -> List[Dict[str, Any]]:
        """Get all audit events associated with a calculation.

        Args:
            calculation_id: Calculation identifier.

        Returns:
            List of event dictionaries ordered by timestamp.
        """
        if not calculation_id or not isinstance(calculation_id, str):
            return []

        with self._lock:
            event_ids = list(self._calculation_index.get(calculation_id, []))

        results: List[Dict[str, Any]] = []
        for eid in event_ids:
            evt = self.get_event(eid)
            if evt is not None:
                results.append(evt)

        # Sort by timestamp ascending
        results.sort(key=lambda e: e.get("timestamp", ""))
        return results

    def get_events_by_scope(
        self,
        organization_id: str,
        reporting_year: int,
        scope: str,
    ) -> List[Dict[str, Any]]:
        """Get all audit events for a specific GHG scope.

        Args:
            organization_id: Organization identifier.
            reporting_year: Reporting year.
            scope: GHG scope (scope_1, scope_2, scope_3).

        Returns:
            List of event dictionaries ordered by chain position.
        """
        self._validate_scope(scope)
        result = self.get_events(
            organization_id=organization_id,
            reporting_year=reporting_year,
            scope=scope,
            limit=_MAX_QUERY_LIMIT,
        )
        return result.get("events", [])

    # =========================================================================
    # PUBLIC API -- Chain Operations
    # =========================================================================

    def get_chain(
        self,
        organization_id: str,
        reporting_year: int,
    ) -> Dict[str, Any]:
        """Get the full audit event chain for an organization/year.

        Args:
            organization_id: Organization identifier.
            reporting_year: Reporting year.

        Returns:
            Dictionary with keys: success, chain_key, genesis_hash,
            head_hash, length, events.
        """
        self._validate_organization_id(organization_id)
        self._validate_reporting_year(reporting_year)

        chain_key = self._get_chain_key(organization_id, reporting_year)

        with self._lock:
            chain = list(self._chains.get(chain_key, []))
            head_hash = self._chain_heads.get(chain_key)

        return {
            "success": True,
            "chain_key": chain_key,
            "organization_id": organization_id,
            "reporting_year": reporting_year,
            "genesis_hash": GENESIS_HASH,
            "head_hash": head_hash,
            "length": len(chain),
            "events": [r.to_dict() for r in chain],
        }

    def verify_chain(
        self,
        organization_id: str,
        reporting_year: int,
        start_position: Optional[int] = None,
        end_position: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Verify the integrity of an audit hash chain.

        Recomputes every hash in the chain (or a sub-range) and confirms
        that each event's ``prev_event_hash`` matches the preceding
        event's ``event_hash``.

        Args:
            organization_id: Organization identifier.
            reporting_year: Reporting year.
            start_position: Optional start position (inclusive, default 0).
            end_position: Optional end position (inclusive, default last).

        Returns:
            Dictionary with keys: success, valid, chain_key,
            verified_count, first_invalid_position, errors,
            verification_time_ms.
        """
        start_ns = time.monotonic_ns()

        self._validate_organization_id(organization_id)
        self._validate_reporting_year(reporting_year)

        chain_key = self._get_chain_key(organization_id, reporting_year)

        with self._lock:
            chain = list(self._chains.get(chain_key, []))

        if not chain:
            elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000
            return {
                "success": True,
                "valid": True,
                "chain_key": chain_key,
                "verified_count": 0,
                "first_invalid_position": None,
                "errors": [],
                "verification_time_ms": round(elapsed_ms, 3),
            }

        # Determine range
        start_pos = start_position if start_position is not None else 0
        end_pos = end_position if end_position is not None else len(chain) - 1
        start_pos = max(0, start_pos)
        end_pos = min(end_pos, len(chain) - 1)

        errors: List[Dict[str, Any]] = []
        first_invalid: Optional[int] = None
        verified = 0

        for i in range(start_pos, end_pos + 1):
            record = chain[i]
            verified += 1

            # Verify prev_hash linkage
            if i == 0:
                expected_prev = GENESIS_HASH
            else:
                expected_prev = chain[i - 1].event_hash

            if record.prev_event_hash != expected_prev:
                error = {
                    "position": i,
                    "event_id": record.event_id,
                    "error": "prev_hash_mismatch",
                    "expected": expected_prev,
                    "actual": record.prev_event_hash,
                }
                errors.append(error)
                if first_invalid is None:
                    first_invalid = i

            # Recompute hash and verify
            recomputed = self._compute_event_hash(
                record.event_id,
                record.prev_event_hash,
                record.event_type,
                record.timestamp,
                record.payload,
            )

            if recomputed != record.event_hash:
                error = {
                    "position": i,
                    "event_id": record.event_id,
                    "error": "hash_mismatch",
                    "expected": recomputed,
                    "actual": record.event_hash,
                }
                errors.append(error)
                if first_invalid is None:
                    first_invalid = i

            # Verify chain_position is consistent
            if record.chain_position != i:
                error = {
                    "position": i,
                    "event_id": record.event_id,
                    "error": "position_mismatch",
                    "expected": i,
                    "actual": record.chain_position,
                }
                errors.append(error)
                if first_invalid is None:
                    first_invalid = i

        is_valid = len(errors) == 0
        elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000

        log_fn = logger.info if is_valid else logger.warning
        log_fn(
            "Chain verification %s | chain=%s | verified=%d | errors=%d | "
            "elapsed=%.3fms",
            "PASSED" if is_valid else "FAILED",
            chain_key,
            verified,
            len(errors),
            elapsed_ms,
        )

        return {
            "success": True,
            "valid": is_valid,
            "chain_key": chain_key,
            "verified_count": verified,
            "first_invalid_position": first_invalid,
            "errors": errors,
            "verification_time_ms": round(elapsed_ms, 3),
        }

    def get_chain_head(
        self,
        organization_id: str,
        reporting_year: int,
    ) -> Optional[str]:
        """Get the latest hash (head) for a chain.

        Args:
            organization_id: Organization identifier.
            reporting_year: Reporting year.

        Returns:
            SHA-256 hex digest of the latest event, or None if the
            chain does not exist.
        """
        chain_key = self._get_chain_key(organization_id, reporting_year)
        with self._lock:
            return self._chain_heads.get(chain_key)

    def get_chain_length(
        self,
        organization_id: str,
        reporting_year: int,
    ) -> int:
        """Get the number of events in a chain.

        Args:
            organization_id: Organization identifier.
            reporting_year: Reporting year.

        Returns:
            Number of events (0 if chain does not exist).
        """
        chain_key = self._get_chain_key(organization_id, reporting_year)
        with self._lock:
            return self._chain_positions.get(chain_key, 0)

    def export_chain(
        self,
        organization_id: str,
        reporting_year: int,
    ) -> Dict[str, Any]:
        """Export a chain for external audit systems.

        Produces a self-contained export including chain metadata,
        all events, and a verification summary.

        Args:
            organization_id: Organization identifier.
            reporting_year: Reporting year.

        Returns:
            Dictionary with keys: success, export_id, chain_key,
            organization_id, reporting_year, genesis_hash, head_hash,
            chain_length, events, verification, exported_at,
            engine_version, agent_id.
        """
        start_ns = time.monotonic_ns()

        chain_data = self.get_chain(organization_id, reporting_year)
        verification = self.verify_chain(organization_id, reporting_year)

        export_id = f"export-{uuid.uuid4().hex[:12]}"
        exported_at = datetime.now(timezone.utc).isoformat()

        elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000

        logger.info(
            "Chain exported | chain=%s | events=%d | valid=%s | "
            "export_id=%s | elapsed=%.3fms",
            chain_data["chain_key"],
            chain_data["length"],
            verification["valid"],
            export_id,
            elapsed_ms,
        )

        return {
            "success": True,
            "export_id": export_id,
            "chain_key": chain_data["chain_key"],
            "organization_id": organization_id,
            "reporting_year": reporting_year,
            "genesis_hash": GENESIS_HASH,
            "head_hash": chain_data["head_hash"],
            "chain_length": chain_data["length"],
            "events": chain_data["events"],
            "verification": {
                "valid": verification["valid"],
                "verified_count": verification["verified_count"],
                "errors": verification["errors"],
            },
            "exported_at": exported_at,
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "agent_id": AGENT_ID,
            "processing_time_ms": round(elapsed_ms, 3),
        }

    # =========================================================================
    # PUBLIC API -- Statistics
    # =========================================================================

    def get_event_statistics(
        self,
        organization_id: str,
        reporting_year: int,
    ) -> Dict[str, Any]:
        """Compute aggregate statistics for a chain.

        Returns counts grouped by event type, scope, agent, and category.

        Args:
            organization_id: Organization identifier.
            reporting_year: Reporting year.

        Returns:
            Dictionary with keys: success, chain_key, total_events,
            by_event_type, by_scope, by_agent, by_category,
            earliest_event, latest_event, avg_data_quality_score.
        """
        self._validate_organization_id(organization_id)
        self._validate_reporting_year(reporting_year)

        chain_key = self._get_chain_key(organization_id, reporting_year)

        with self._lock:
            chain = list(self._chains.get(chain_key, []))

        if not chain:
            return {
                "success": True,
                "chain_key": chain_key,
                "total_events": 0,
                "by_event_type": {},
                "by_scope": {},
                "by_agent": {},
                "by_category": {},
                "earliest_event": None,
                "latest_event": None,
                "avg_data_quality_score": None,
            }

        by_type: Dict[str, int] = defaultdict(int)
        by_scope: Dict[str, int] = defaultdict(int)
        by_agent: Dict[str, int] = defaultdict(int)
        by_category: Dict[str, int] = defaultdict(int)
        dq_sum = _ZERO
        dq_count = 0

        for record in chain:
            by_type[record.event_type] += 1
            if record.scope is not None:
                by_scope[record.scope] += 1
            by_agent[record.agent_id] += 1
            if record.category is not None:
                by_category[str(record.category)] += 1
            dq_sum += record.data_quality_score
            dq_count += 1

        avg_dq: Optional[str] = None
        if dq_count > 0:
            avg_dq = str(
                (dq_sum / dq_count).quantize(_QUANT_4DP, rounding=ROUND_HALF_UP)
            )

        return {
            "success": True,
            "chain_key": chain_key,
            "total_events": len(chain),
            "by_event_type": dict(by_type),
            "by_scope": dict(by_scope),
            "by_agent": dict(by_agent),
            "by_category": dict(by_category),
            "earliest_event": chain[0].timestamp,
            "latest_event": chain[-1].timestamp,
            "avg_data_quality_score": avg_dq,
        }

    # =========================================================================
    # PUBLIC API -- Reset (Testing Only)
    # =========================================================================

    def reset(self) -> None:
        """Reset all engine state.

        Intended for unit and integration testing only.  This clears
        every chain, index, and counter.

        WARNING: This destroys all recorded audit events.
        """
        with self._lock:
            self._chains.clear()
            self._chain_heads.clear()
            self._chain_positions.clear()
            self._event_index.clear()
            self._calculation_index.clear()

        logger.warning(
            "AuditEventEngine reset | engine=%s | ALL DATA CLEARED",
            ENGINE_ID,
        )

    # =========================================================================
    # INTERNAL -- Event Recording (Lock-free, caller holds lock)
    # =========================================================================

    def _record_event_internal(
        self,
        event_type: str,
        agent_id: str,
        scope: Optional[str],
        category: Optional[int],
        organization_id: str,
        reporting_year: int,
        calculation_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        data_quality_score: Optional[Decimal] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record an event without acquiring the lock.

        This internal method is called by ``record_batch`` which already
        holds the lock.  It performs the same logic as ``record_event``
        but skips lock acquisition.

        Args:
            (Same as record_event.)

        Returns:
            Dictionary with event details.
        """
        safe_payload = payload if payload is not None else {}
        safe_metadata = metadata if metadata is not None else {}
        dq_score = self._normalize_dq_score(data_quality_score)

        event_id = self._generate_event_id()
        timestamp = datetime.now(timezone.utc).isoformat()
        chain_key = self._get_chain_key(organization_id, reporting_year)

        prev_hash = self._chain_heads.get(chain_key, GENESIS_HASH)
        position = self._chain_positions.get(chain_key, 0)

        event_hash = self._compute_event_hash(
            event_id, prev_hash, event_type, timestamp, safe_payload,
        )

        record = AuditEventRecord(
            event_id=event_id,
            event_type=event_type,
            agent_id=agent_id,
            scope=scope,
            category=category,
            organization_id=organization_id,
            reporting_year=reporting_year,
            calculation_id=calculation_id,
            data_quality_score=dq_score,
            payload=safe_payload,
            prev_event_hash=prev_hash,
            event_hash=event_hash,
            chain_position=position,
            timestamp=timestamp,
            metadata=safe_metadata,
        )

        if chain_key not in self._chains:
            self._chains[chain_key] = []
        self._chains[chain_key].append(record)
        self._chain_heads[chain_key] = event_hash
        self._chain_positions[chain_key] = position + 1

        self._event_index[event_id] = record
        if calculation_id:
            self._calculation_index[calculation_id].append(event_id)

        return {
            "success": True,
            "event_id": event_id,
            "event_type": event_type,
            "event_hash": event_hash,
            "prev_event_hash": prev_hash,
            "chain_position": position,
            "chain_key": chain_key,
            "timestamp": timestamp,
        }

    # =========================================================================
    # INTERNAL -- Hashing (ZERO HALLUCINATION)
    # =========================================================================

    def _compute_event_hash(
        self,
        event_id: str,
        prev_hash: str,
        event_type: str,
        timestamp: str,
        payload: Dict[str, Any],
    ) -> str:
        """Compute SHA-256 hash for an audit event.

        The hash input is a pipe-delimited string:
            ``{event_id}|{prev_hash}|{event_type}|{timestamp}|{canonical_json(payload)}``

        This is a deterministic, zero-hallucination operation:
        no LLM or ML model is involved.

        Args:
            event_id: Unique event identifier.
            prev_hash: Hash of the previous event in the chain.
            event_type: Audit event type.
            timestamp: ISO 8601 UTC timestamp.
            payload: Event payload dictionary.

        Returns:
            64-character lowercase hex SHA-256 digest.
        """
        canonical = self._canonical_json(payload)
        hash_input = f"{event_id}|{prev_hash}|{event_type}|{timestamp}|{canonical}"
        return hashlib.sha256(hash_input.encode(ENCODING)).hexdigest()

    def _canonical_json(self, data: Any) -> str:
        """Produce deterministic JSON for hashing.

        Keys are sorted, no whitespace, and special types (Decimal,
        datetime, Enum) are serialized via ``_decimal_serializer``.

        Args:
            data: Data to serialize.

        Returns:
            Canonical JSON string.
        """
        return json.dumps(
            data,
            sort_keys=True,
            separators=(",", ":"),
            default=_decimal_serializer,
        )

    # =========================================================================
    # INTERNAL -- Validation Helpers
    # =========================================================================

    def _validate_event_type(self, event_type: str) -> bool:
        """Validate event_type against the 12 known types.

        Args:
            event_type: Event type string to validate.

        Returns:
            True if valid.

        Raises:
            ValueError: If event_type is not recognized.
        """
        if not event_type or not isinstance(event_type, str):
            raise ValueError("event_type must be a non-empty string")
        if event_type not in _VALID_EVENT_TYPES:
            raise ValueError(
                f"Invalid event_type '{event_type}'. "
                f"Must be one of: {sorted(_VALID_EVENT_TYPES)}"
            )
        return True

    def _validate_organization_id(self, organization_id: str) -> None:
        """Validate that organization_id is a non-empty string.

        Args:
            organization_id: Organization identifier.

        Raises:
            ValueError: If organization_id is empty or not a string.
        """
        if not organization_id or not isinstance(organization_id, str):
            raise ValueError("organization_id must be a non-empty string")

    def _validate_reporting_year(self, reporting_year: int) -> None:
        """Validate that reporting_year is a reasonable integer.

        Args:
            reporting_year: Reporting year to validate.

        Raises:
            ValueError: If year is out of range [1990, 2100].
        """
        if not isinstance(reporting_year, int):
            raise ValueError("reporting_year must be an integer")
        if reporting_year < 1990 or reporting_year > 2100:
            raise ValueError(
                f"reporting_year {reporting_year} out of range [1990, 2100]"
            )

    def _validate_scope(self, scope: Optional[str]) -> None:
        """Validate GHG scope value.

        Args:
            scope: GHG scope string or None.

        Raises:
            ValueError: If scope is provided but not one of the valid values.
        """
        if scope is not None:
            if not isinstance(scope, str):
                raise ValueError("scope must be a string or None")
            if scope not in _VALID_SCOPES:
                raise ValueError(
                    f"Invalid scope '{scope}'. "
                    f"Must be one of: {sorted(_VALID_SCOPES)}"
                )

    def _validate_category(self, category: Optional[int]) -> None:
        """Validate Scope 3 category number.

        Args:
            category: Scope 3 category (1-15) or None.

        Raises:
            ValueError: If category is provided but out of range.
        """
        if category is not None:
            if not isinstance(category, int):
                raise ValueError("category must be an integer or None")
            if category not in _VALID_CATEGORIES:
                raise ValueError(
                    f"Invalid category {category}. Must be in range [1, 15]"
                )

    def _validate_agent_id(self, agent_id: str) -> None:
        """Validate that agent_id is a non-empty string.

        Args:
            agent_id: Agent identifier.

        Raises:
            ValueError: If agent_id is empty or not a string.
        """
        if not agent_id or not isinstance(agent_id, str):
            raise ValueError("agent_id must be a non-empty string")

    def _validate_batch_event_dict(
        self, evt: Dict[str, Any], idx: int,
    ) -> None:
        """Validate a single event dictionary in a batch.

        Checks that all required keys are present and validates their
        values using the individual validators.

        Args:
            evt: Event dictionary.
            idx: Index in the batch (for error messages).

        Raises:
            ValueError: If any required field is missing or invalid.
            TypeError: If evt is not a dictionary.
        """
        if not isinstance(evt, dict):
            raise TypeError(f"Event at index {idx} must be a dictionary")

        required_keys = {
            "event_type",
            "agent_id",
            "organization_id",
            "reporting_year",
        }
        missing = required_keys - set(evt.keys())
        if missing:
            raise ValueError(
                f"Event at index {idx} missing required keys: {sorted(missing)}"
            )

        self._validate_event_type(evt["event_type"])
        self._validate_agent_id(evt["agent_id"])
        self._validate_organization_id(evt["organization_id"])
        self._validate_reporting_year(evt["reporting_year"])
        self._validate_scope(evt.get("scope"))
        self._validate_category(evt.get("category"))

    # =========================================================================
    # INTERNAL -- Utility Helpers
    # =========================================================================

    def _normalize_dq_score(
        self, score: Optional[Decimal],
    ) -> Decimal:
        """Normalize data quality score to Decimal in [0.00, 1.00].

        Accepts Decimal, int, float, or string.  Clamps to [0, 1] and
        quantizes to 2 decimal places.

        Args:
            score: Raw data quality score or None (defaults to 0.00).

        Returns:
            Decimal quantized to 2 decimal places.
        """
        if score is None:
            return _ZERO

        try:
            if isinstance(score, Decimal):
                d = score
            elif isinstance(score, (int, float)):
                d = Decimal(str(score))
            elif isinstance(score, str):
                d = Decimal(score)
            else:
                d = _ZERO
        except (InvalidOperation, ValueError):
            d = _ZERO

        # Clamp to [0, 1]
        if d < _ZERO:
            d = _ZERO
        elif d > _ONE:
            d = _ONE

        return d.quantize(_QUANT_2DP, rounding=ROUND_HALF_UP)

    def _generate_event_id(self) -> str:
        """Generate a unique event identifier.

        Returns:
            UUID v4 string prefixed with ``atl-``.
        """
        return f"atl-{uuid.uuid4().hex}"

    def _get_chain_key(
        self, organization_id: str, reporting_year: int,
    ) -> str:
        """Build the chain storage key.

        Args:
            organization_id: Organization identifier.
            reporting_year: Reporting year.

        Returns:
            Key string in format ``{organization_id}:{reporting_year}``.
        """
        return f"{organization_id}:{reporting_year}"

    def _apply_filters(
        self,
        chain: List[AuditEventRecord],
        event_type: Optional[str] = None,
        agent_id: Optional[str] = None,
        scope: Optional[str] = None,
        category: Optional[int] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> List[AuditEventRecord]:
        """Apply query filters to a chain of events.

        Each filter is an AND condition: only events matching all
        non-None filters are included.

        Args:
            chain: List of events to filter.
            event_type: Filter by event type.
            agent_id: Filter by agent identifier.
            scope: Filter by GHG scope.
            category: Filter by Scope 3 category.
            start_time: ISO 8601 lower bound (inclusive).
            end_time: ISO 8601 upper bound (inclusive).

        Returns:
            Filtered list of AuditEventRecord instances.
        """
        result = chain

        if event_type is not None:
            result = [r for r in result if r.event_type == event_type]

        if agent_id is not None:
            result = [r for r in result if r.agent_id == agent_id]

        if scope is not None:
            result = [r for r in result if r.scope == scope]

        if category is not None:
            result = [r for r in result if r.category == category]

        if start_time is not None:
            result = [r for r in result if r.timestamp >= start_time]

        if end_time is not None:
            result = [r for r in result if r.timestamp <= end_time]

        return result


# =============================================================================
# MODULE-LEVEL ACCESSOR
# =============================================================================


def get_audit_event_engine() -> AuditEventEngine:
    """Get the singleton AuditEventEngine instance.

    This is the recommended entry point for obtaining the engine.

    Returns:
        The singleton AuditEventEngine instance.

    Example:
        >>> engine = get_audit_event_engine()
        >>> result = engine.record_event(...)
    """
    return AuditEventEngine()
