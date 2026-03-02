# -*- coding: utf-8 -*-
"""
Audit Trail & Lineage Provenance Tracking - AGENT-MRV-030

SHA-256 chain-hashed provenance for zero-hallucination audit trails.
12 pipeline stages tracked with deterministic hashing.
Agent: GL-MRV-X-042

This module implements a complete provenance tracking system for the
Audit Trail & Lineage agent. Every step in the audit trail pipeline
is recorded with SHA-256 hashes, creating an immutable, verifiable
chain that proves no audit event was hallucinated, tampered with, or
silently modified.

Pipeline Stages:
    1.  INGEST       - Raw event ingestion
    2.  VALIDATE     - Input validation and normalization
    3.  CLASSIFY     - Scope/category/framework classification
    4.  RECORD       - Immutable hash chain recording
    5.  LINK         - Lineage graph node linking
    6.  TRACE        - Regulatory requirement tracing
    7.  DETECT       - Change detection
    8.  VERIFY       - Hash chain verification
    9.  PACKAGE      - Evidence packaging
    10. COMPLIANCE   - Compliance checking
    11. SEAL         - Final provenance seal
    12. COMPLETE     - Pipeline completion

16 Hash Functions:
    1.  hash_audit_event()           - Hash raw audit event data
    2.  hash_lineage_node()          - Hash lineage graph node
    3.  hash_lineage_edge()          - Hash lineage graph edge
    4.  hash_evidence_package()      - Hash evidence package bundle
    5.  hash_compliance_trace()      - Hash compliance trace mapping
    6.  hash_change_event()          - Hash change detection result
    7.  hash_chain_verification()    - Hash chain verification result
    8.  hash_pipeline_result()       - Hash complete pipeline result
    9.  hash_classification()        - Hash event classification
    10. hash_seal()                  - Hash final seal data
    11. hash_batch_result()          - Hash batch pipeline result
    12. hash_stage_result()          - Hash individual stage result
    13. hash_event_payload()         - Hash event payload data
    14. hash_framework_mapping()     - Hash framework requirement mapping
    15. hash_recalculation()         - Hash recalculation trigger
    16. build_chain_hash()           - SHA-256(P1 || P2 || ... || Pn)

Example:
    >>> tracker = ProvenanceTracker()
    >>> chain_id = tracker.start_chain()
    >>> tracker.record_stage(
    ...     chain_id, ProvenanceStage.VALIDATE,
    ...     {"event_type": "calculation_completed"},
    ...     {"validated": True},
    ... )
    >>> sealed = tracker.seal_chain(chain_id)
    >>> assert tracker.verify_chain(chain_id)["valid"]

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-X-042
Date: March 2026
"""

import hashlib
import json
import logging
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

AGENT_ID = "GL-MRV-X-042"
AGENT_VERSION = "1.0.0"
HASH_ALGORITHM = "sha256"
ENCODING = "utf-8"


class ProvenanceStage(str, Enum):
    """Pipeline stages for audit trail provenance tracking (12 stages)."""

    INGEST = "INGEST"
    VALIDATE = "VALIDATE"
    CLASSIFY = "CLASSIFY"
    RECORD = "RECORD"
    LINK = "LINK"
    TRACE = "TRACE"
    DETECT = "DETECT"
    VERIFY = "VERIFY"
    PACKAGE = "PACKAGE"
    COMPLIANCE = "COMPLIANCE"
    SEAL = "SEAL"
    COMPLETE = "COMPLETE"


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass(frozen=True)
class ProvenanceEntry:
    """
    Single provenance entry in a chain.

    Frozen (immutable) record of one stage in the audit trail pipeline.
    The chain_hash links this entry to the previous entry, forming an
    immutable hash chain.

    Attributes:
        entry_id: Unique identifier for this entry (UUID).
        stage: Pipeline stage name (from ProvenanceStage enum).
        timestamp: ISO 8601 UTC timestamp of entry creation.
        input_hash: SHA-256 hash of input data to this stage.
        output_hash: SHA-256 hash of output data from this stage.
        chain_hash: SHA-256 hash combining stage + input_hash + output_hash + previous_hash.
        previous_hash: Chain hash from the previous entry (empty for first entry).
        agent_id: Agent identifier (GL-MRV-X-042).
        metadata: Additional context (optional).
    """

    entry_id: str
    stage: str
    timestamp: str
    input_hash: str
    output_hash: str
    chain_hash: str
    previous_hash: str
    agent_id: str = AGENT_ID
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert entry to JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True)


@dataclass
class ProvenanceChain:
    """
    Complete provenance chain for an audit trail pipeline run.

    Tracks all entries in pipeline order with chain-level hash.

    Attributes:
        chain_id: Unique identifier for this chain.
        agent_id: Agent identifier (GL-MRV-X-042).
        entries: Ordered list of provenance entries.
        sealed: Whether the chain has been sealed (finalized).
        final_hash: Final chain hash after sealing (None if not sealed).
    """

    chain_id: str
    agent_id: str = AGENT_ID
    entries: List[ProvenanceEntry] = field(default_factory=list)
    sealed: bool = False
    final_hash: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def is_valid(self) -> bool:
        """Whether the chain has entries and a computed final hash."""
        return len(self.entries) > 0 and self.sealed and self.final_hash is not None

    @property
    def entry_count(self) -> int:
        """Number of entries recorded."""
        return len(self.entries)

    @property
    def stages_recorded(self) -> List[str]:
        """List of stage names recorded in order."""
        return [e.stage for e in self.entries]

    def to_dict(self) -> Dict[str, Any]:
        """Convert chain to dictionary."""
        return {
            "chain_id": self.chain_id,
            "agent_id": self.agent_id,
            "entries": [e.to_dict() for e in self.entries],
            "sealed": self.sealed,
            "final_hash": self.final_hash,
            "created_at": self.created_at,
            "is_valid": self.is_valid,
            "entry_count": self.entry_count,
            "stages_recorded": self.stages_recorded,
        }

    def to_json(self) -> str:
        """Convert chain to JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


# ============================================================================
# SERIALIZATION AND HASH UTILITIES
# ============================================================================


def _serialize(data: Any) -> str:
    """
    Serialize object to deterministic JSON string.

    Converts Decimal to string, datetime to ISO format, Enum to value,
    sorts keys, and handles nested dicts/lists and Pydantic models.

    Args:
        data: Object to serialize.

    Returns:
        Deterministic JSON string.
    """

    def default_handler(o: Any) -> Any:
        """Handle non-JSON-serializable types."""
        if isinstance(o, Decimal):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, Enum):
            return o.value
        if hasattr(o, "model_dump"):
            return o.model_dump()
        if hasattr(o, "dict"):
            return o.dict()
        if hasattr(o, "to_dict"):
            return o.to_dict()
        if hasattr(o, "__dict__"):
            return o.__dict__
        return str(o)

    return json.dumps(data, sort_keys=True, default=default_handler)


def _compute_hash(data: Any) -> str:
    """
    Compute SHA-256 hash of data.

    Args:
        data: Data to hash (any JSON-serializable type).

    Returns:
        Lowercase hex SHA-256 hash.
    """
    serialized = _serialize(data)
    return hashlib.sha256(serialized.encode(ENCODING)).hexdigest()


def _compute_stage_hash(
    stage: str, input_hash: str, output_hash: str, previous_hash: str
) -> str:
    """
    Compute chain hash combining stage name with input, output, and previous hashes.

    Chain hash = SHA-256(stage + "|" + input_hash + "|" + output_hash + "|" + previous_hash)

    Args:
        stage: Pipeline stage name.
        input_hash: Hash of input data.
        output_hash: Hash of output data.
        previous_hash: Hash from previous chain entry.

    Returns:
        SHA-256 chain hash.
    """
    chain_data = f"{stage}|{input_hash}|{output_hash}|{previous_hash}"
    return hashlib.sha256(chain_data.encode(ENCODING)).hexdigest()


def _compute_chain_hash(entry_hashes: List[str]) -> str:
    """
    Compute final chain hash from ordered list of entry chain hashes.

    Chain hash = SHA-256(hash_1 || hash_2 || ... || hash_n)

    Args:
        entry_hashes: Ordered list of per-entry chain hashes.

    Returns:
        SHA-256 chain hash.
    """
    if not entry_hashes:
        return hashlib.sha256(b"").hexdigest()
    combined = "||".join(entry_hashes)
    return hashlib.sha256(combined.encode(ENCODING)).hexdigest()


# ============================================================================
# PROVENANCE TRACKER (THREAD-SAFE SINGLETON)
# ============================================================================


class ProvenanceTracker:
    """
    Provenance tracking engine for the Audit Trail & Lineage agent.

    Manages creation, chaining, sealing, and verification of provenance
    records across the 12-stage audit trail pipeline. Thread-safe singleton
    with RLock protection on all mutable state.

    Attributes:
        agent_id: Agent identifier.
        agent_version: Agent version string.

    Example:
        >>> tracker = ProvenanceTracker()
        >>> chain_id = tracker.start_chain()
        >>> tracker.record_stage(
        ...     chain_id, ProvenanceStage.VALIDATE,
        ...     {"event_type": "calculation_completed"},
        ...     {"validated": True},
        ... )
        >>> sealed = tracker.seal_chain(chain_id)
        >>> assert sealed["sealed"]
        >>> verify = tracker.verify_chain(chain_id)
        >>> assert verify["valid"]
    """

    _instance: Optional["ProvenanceTracker"] = None
    _instance_lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "ProvenanceTracker":
        """Thread-safe singleton instantiation."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the provenance tracker (only once due to singleton)."""
        if hasattr(self, "_initialized"):
            return

        self._initialized: bool = True
        self.agent_id: str = AGENT_ID
        self.agent_version: str = AGENT_VERSION
        self._lock: threading.RLock = threading.RLock()
        self._chains: Dict[str, ProvenanceChain] = {}
        self._entries: Dict[str, ProvenanceEntry] = {}

        logger.info(
            "ProvenanceTracker initialized: agent_id=%s, version=%s",
            self.agent_id,
            self.agent_version,
        )

    # ------------------------------------------------------------------
    # Chain lifecycle
    # ------------------------------------------------------------------

    def start_chain(self, chain_id: Optional[str] = None) -> str:
        """
        Start a new provenance chain.

        Creates a new empty chain that entries can be recorded against.

        Args:
            chain_id: Optional chain identifier. If None, a UUID is generated.

        Returns:
            Chain identifier string.

        Raises:
            ValueError: If a chain with the given ID already exists.
        """
        with self._lock:
            if chain_id is None:
                chain_id = f"prov-{uuid.uuid4().hex[:12]}"

            if chain_id in self._chains:
                raise ValueError(
                    f"Chain '{chain_id}' already exists. "
                    "Use a unique chain_id or omit for auto-generation."
                )

            chain = ProvenanceChain(
                chain_id=chain_id,
                agent_id=self.agent_id,
            )
            self._chains[chain_id] = chain

            logger.debug("Started provenance chain: %s", chain_id)
            return chain_id

    def record_stage(
        self,
        chain_id: str,
        stage: Union[ProvenanceStage, str],
        input_data: Any,
        output_data: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Record a pipeline stage in the provenance chain.

        Computes SHA-256 hashes of input and output data, derives a
        chain hash linked to the previous entry, and appends an
        immutable entry to the chain.

        Args:
            chain_id: Chain to record against.
            stage: Pipeline stage identifier.
            input_data: Input data to this stage.
            output_data: Output data from this stage.
            metadata: Optional metadata dictionary.

        Returns:
            Entry identifier string.

        Raises:
            ValueError: If chain does not exist or is already sealed.
        """
        with self._lock:
            chain = self._chains.get(chain_id)
            if chain is None:
                raise ValueError(f"Chain '{chain_id}' does not exist")
            if chain.sealed:
                raise ValueError(
                    f"Chain '{chain_id}' is already sealed; "
                    "no further entries can be recorded"
                )

            stage_str = (
                stage.value if isinstance(stage, ProvenanceStage) else str(stage)
            )
            input_hash = _compute_hash(input_data)
            output_hash = _compute_hash(output_data)

            # Link to previous entry
            previous_hash = ""
            if chain.entries:
                previous_hash = chain.entries[-1].chain_hash

            chain_hash = _compute_stage_hash(
                stage_str, input_hash, output_hash, previous_hash
            )

            entry_id = f"entry-{uuid.uuid4().hex[:12]}"

            entry = ProvenanceEntry(
                entry_id=entry_id,
                stage=stage_str,
                timestamp=datetime.now(timezone.utc).isoformat(),
                input_hash=input_hash,
                output_hash=output_hash,
                chain_hash=chain_hash,
                previous_hash=previous_hash,
                agent_id=self.agent_id,
                metadata=metadata or {},
            )

            chain.entries.append(entry)
            self._entries[entry_id] = entry

            logger.debug(
                "Recorded stage %s in chain %s: entry_id=%s",
                stage_str,
                chain_id,
                entry_id,
            )
            return entry_id

    def seal_chain(self, chain_id: str) -> Dict[str, Any]:
        """
        Seal a provenance chain, computing the final hash.

        Once sealed, no further entries can be recorded. The final hash
        covers all entry chain hashes in order.

        Args:
            chain_id: Chain to seal.

        Returns:
            Dictionary with chain_id, sealed status, final_hash,
            entry_count, and stages_recorded.

        Raises:
            ValueError: If chain does not exist or is already sealed.
        """
        with self._lock:
            chain = self._chains.get(chain_id)
            if chain is None:
                raise ValueError(f"Chain '{chain_id}' does not exist")
            if chain.sealed:
                raise ValueError(
                    f"Chain '{chain_id}' is already sealed"
                )
            if not chain.entries:
                raise ValueError(
                    f"Chain '{chain_id}' has no entries to seal"
                )

            entry_hashes = [e.chain_hash for e in chain.entries]
            final_hash = _compute_chain_hash(entry_hashes)

            chain.sealed = True
            chain.final_hash = final_hash

            logger.info(
                "Sealed provenance chain %s: final_hash=%s..., entries=%d",
                chain_id,
                final_hash[:16],
                len(chain.entries),
            )

            return {
                "chain_id": chain_id,
                "sealed": True,
                "final_hash": final_hash,
                "entry_count": chain.entry_count,
                "stages_recorded": chain.stages_recorded,
            }

    def verify_chain(self, chain_id: str) -> Dict[str, Any]:
        """
        Verify integrity of a provenance chain.

        Checks:
        1. Each entry's chain_hash matches recomputed value.
        2. Each entry's previous_hash matches the prior entry's chain_hash.
        3. If sealed, the final_hash matches recomputation.

        Args:
            chain_id: Chain to verify.

        Returns:
            Dictionary with chain_id, valid status, entry_count,
            verification_hash, and any errors found.

        Raises:
            ValueError: If chain does not exist.
        """
        with self._lock:
            chain = self._chains.get(chain_id)
            if chain is None:
                raise ValueError(f"Chain '{chain_id}' does not exist")

            if not chain.entries:
                return {
                    "chain_id": chain_id,
                    "valid": True,
                    "entry_count": 0,
                    "verification_hash": "",
                    "errors": [],
                }

            errors: List[str] = []

            # Verify each entry
            for idx, entry in enumerate(chain.entries):
                # Check chain_hash
                expected_previous = (
                    chain.entries[idx - 1].chain_hash if idx > 0 else ""
                )
                expected_chain_hash = _compute_stage_hash(
                    entry.stage,
                    entry.input_hash,
                    entry.output_hash,
                    expected_previous,
                )

                if entry.chain_hash != expected_chain_hash:
                    errors.append(
                        f"Entry {idx} ({entry.stage}): chain_hash mismatch"
                    )

                # Check previous_hash linkage
                if entry.previous_hash != expected_previous:
                    errors.append(
                        f"Entry {idx} ({entry.stage}): previous_hash mismatch"
                    )

            # Verify final hash if sealed
            if chain.sealed and chain.final_hash is not None:
                entry_hashes = [e.chain_hash for e in chain.entries]
                expected_final = _compute_chain_hash(entry_hashes)
                if chain.final_hash != expected_final:
                    errors.append("Final chain hash mismatch")

            valid = len(errors) == 0
            verification_hash = _compute_hash({
                "chain_id": chain_id,
                "entry_count": chain.entry_count,
                "valid": valid,
                "verified_at": datetime.now(timezone.utc).isoformat(),
            })

            if not valid:
                logger.warning(
                    "Chain %s verification FAILED: %d errors",
                    chain_id,
                    len(errors),
                )

            return {
                "chain_id": chain_id,
                "valid": valid,
                "entry_count": chain.entry_count,
                "verification_hash": verification_hash,
                "errors": errors,
            }

    # ------------------------------------------------------------------
    # Chain retrieval
    # ------------------------------------------------------------------

    def get_chain(self, chain_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a provenance chain by ID.

        Args:
            chain_id: Chain identifier.

        Returns:
            Chain dictionary, or None if not found.
        """
        with self._lock:
            chain = self._chains.get(chain_id)
            if chain is None:
                return None
            return chain.to_dict()

    def get_entry(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a provenance entry by ID.

        Args:
            entry_id: Entry identifier.

        Returns:
            Entry dictionary, or None if not found.
        """
        with self._lock:
            entry = self._entries.get(entry_id)
            if entry is None:
                return None
            return entry.to_dict()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Reset all chains and entries.

        Clears all in-memory provenance data. Primarily for testing.
        """
        with self._lock:
            self._chains.clear()
            self._entries.clear()
            logger.info("ProvenanceTracker state reset")

    @classmethod
    def reset_singleton(cls) -> None:
        """
        Reset the singleton instance.

        Creates a fresh instance on next access.
        """
        with cls._instance_lock:
            cls._instance = None
            logger.info("ProvenanceTracker singleton reset")


# ============================================================================
# STANDALONE HASH FUNCTIONS (16)
# ============================================================================


def hash_audit_event(
    event_type: str,
    agent_id: str,
    scope: str,
    category: str,
    calculation_id: str,
    payload: Dict[str, Any],
) -> str:
    """
    Hash raw audit event data (Hash Function 1).

    Args:
        event_type: Type of audit event.
        agent_id: Source agent identifier.
        scope: GHG emission scope.
        category: Emission category.
        calculation_id: Calculation identifier.
        payload: Event payload data.

    Returns:
        SHA-256 hash.
    """
    data = {
        "event_type": event_type,
        "agent_id": agent_id,
        "scope": scope,
        "category": category,
        "calculation_id": calculation_id,
        "payload": payload,
    }
    return _compute_hash(data)


def hash_lineage_node(
    node_id: str,
    node_type: str,
    agent_id: str,
    calculation_id: str,
    attributes: Dict[str, Any],
) -> str:
    """
    Hash lineage graph node (Hash Function 2).

    Args:
        node_id: Node identifier.
        node_type: Node type (source, calculation, output, etc.).
        agent_id: Agent that created the node.
        calculation_id: Associated calculation.
        attributes: Node attributes.

    Returns:
        SHA-256 hash.
    """
    data = {
        "node_id": node_id,
        "node_type": node_type,
        "agent_id": agent_id,
        "calculation_id": calculation_id,
        "attributes": attributes,
    }
    return _compute_hash(data)


def hash_lineage_edge(
    source_node_id: str,
    target_node_id: str,
    edge_type: str,
    attributes: Dict[str, Any],
) -> str:
    """
    Hash lineage graph edge (Hash Function 3).

    Args:
        source_node_id: Source node identifier.
        target_node_id: Target node identifier.
        edge_type: Edge type (derived_from, depends_on, etc.).
        attributes: Edge attributes.

    Returns:
        SHA-256 hash.
    """
    data = {
        "source_node_id": source_node_id,
        "target_node_id": target_node_id,
        "edge_type": edge_type,
        "attributes": attributes,
    }
    return _compute_hash(data)


def hash_evidence_package(
    package_id: str,
    event_ids: List[str],
    chain_hashes: List[str],
    completeness_score: float,
) -> str:
    """
    Hash evidence package bundle (Hash Function 4).

    Args:
        package_id: Evidence package identifier.
        event_ids: List of event IDs included.
        chain_hashes: List of chain hashes included.
        completeness_score: Evidence completeness score.

    Returns:
        SHA-256 hash.
    """
    data = {
        "package_id": package_id,
        "event_ids": sorted(event_ids),
        "chain_hashes": sorted(chain_hashes),
        "completeness_score": str(completeness_score),
    }
    return _compute_hash(data)


def hash_compliance_trace(
    framework: str,
    requirement_id: str,
    event_id: str,
    trace_details: Dict[str, Any],
) -> str:
    """
    Hash compliance trace mapping (Hash Function 5).

    Args:
        framework: Regulatory framework identifier.
        requirement_id: Framework requirement identifier.
        event_id: Audit event identifier.
        trace_details: Trace details.

    Returns:
        SHA-256 hash.
    """
    data = {
        "framework": framework,
        "requirement_id": requirement_id,
        "event_id": event_id,
        "trace_details": trace_details,
    }
    return _compute_hash(data)


def hash_change_event(
    calculation_id: str,
    change_type: str,
    old_value: Any,
    new_value: Any,
    materiality: str,
) -> str:
    """
    Hash change detection result (Hash Function 6).

    Args:
        calculation_id: Affected calculation identifier.
        change_type: Type of change detected.
        old_value: Previous value.
        new_value: New value.
        materiality: Materiality classification (material/immaterial).

    Returns:
        SHA-256 hash.
    """
    data = {
        "calculation_id": calculation_id,
        "change_type": change_type,
        "old_value": str(old_value),
        "new_value": str(new_value),
        "materiality": materiality,
    }
    return _compute_hash(data)


def hash_chain_verification(
    chain_id: str,
    valid: bool,
    entry_count: int,
    errors: List[str],
) -> str:
    """
    Hash chain verification result (Hash Function 7).

    Args:
        chain_id: Chain identifier verified.
        valid: Whether chain passed verification.
        entry_count: Number of entries in the chain.
        errors: List of verification errors found.

    Returns:
        SHA-256 hash.
    """
    data = {
        "chain_id": chain_id,
        "valid": valid,
        "entry_count": entry_count,
        "errors": sorted(errors),
    }
    return _compute_hash(data)


def hash_pipeline_result(
    run_id: str,
    status: str,
    event_id: str,
    seal_hash: str,
    stage_count: int,
) -> str:
    """
    Hash complete pipeline result (Hash Function 8).

    Args:
        run_id: Pipeline run identifier.
        status: Pipeline outcome status.
        event_id: Generated event identifier.
        seal_hash: Final seal hash.
        stage_count: Number of stages completed.

    Returns:
        SHA-256 hash.
    """
    data = {
        "run_id": run_id,
        "status": status,
        "event_id": event_id,
        "seal_hash": seal_hash,
        "stage_count": stage_count,
    }
    return _compute_hash(data)


def hash_classification(
    scope: str,
    category: str,
    frameworks: List[str],
    priority: str,
) -> str:
    """
    Hash event classification (Hash Function 9).

    Args:
        scope: GHG emission scope.
        category: Emission category.
        frameworks: Applicable regulatory frameworks.
        priority: Event priority (normal/high).

    Returns:
        SHA-256 hash.
    """
    data = {
        "scope": scope,
        "category": category,
        "frameworks": sorted(frameworks),
        "priority": priority,
    }
    return _compute_hash(data)


def hash_seal(
    run_id: str,
    chain_hash: str,
    stage_durations_ms: Dict[str, float],
    timestamp: str,
) -> str:
    """
    Hash final seal data (Hash Function 10).

    Args:
        run_id: Pipeline run identifier.
        chain_hash: Accumulated chain hash.
        stage_durations_ms: Per-stage durations.
        timestamp: Seal timestamp.

    Returns:
        SHA-256 hash.
    """
    data = {
        "run_id": run_id,
        "chain_hash": chain_hash,
        "stage_durations_ms": stage_durations_ms,
        "timestamp": timestamp,
    }
    return _compute_hash(data)


def hash_batch_result(
    batch_id: str,
    total_events: int,
    successes: int,
    failures: int,
    processing_time_ms: float,
) -> str:
    """
    Hash batch pipeline result (Hash Function 11).

    Args:
        batch_id: Batch identifier.
        total_events: Total events in batch.
        successes: Number of successful events.
        failures: Number of failed events.
        processing_time_ms: Total batch processing time.

    Returns:
        SHA-256 hash.
    """
    data = {
        "batch_id": batch_id,
        "total_events": total_events,
        "successes": successes,
        "failures": failures,
        "processing_time_ms": str(processing_time_ms),
    }
    return _compute_hash(data)


def hash_stage_result(
    stage: str,
    status: str,
    duration_ms: float,
    data_snapshot: Dict[str, Any],
) -> str:
    """
    Hash individual stage result (Hash Function 12).

    Args:
        stage: Pipeline stage name.
        status: Stage outcome status.
        duration_ms: Stage duration in milliseconds.
        data_snapshot: Key stage output data.

    Returns:
        SHA-256 hash.
    """
    data = {
        "stage": stage,
        "status": status,
        "duration_ms": str(duration_ms),
        "data_snapshot": data_snapshot,
    }
    return _compute_hash(data)


def hash_event_payload(
    calculation_id: str,
    payload: Dict[str, Any],
    data_quality_score: float,
) -> str:
    """
    Hash event payload data (Hash Function 13).

    Args:
        calculation_id: Calculation identifier.
        payload: Event payload.
        data_quality_score: Data quality score (0.0-1.0).

    Returns:
        SHA-256 hash.
    """
    data = {
        "calculation_id": calculation_id,
        "payload": payload,
        "data_quality_score": str(data_quality_score),
    }
    return _compute_hash(data)


def hash_framework_mapping(
    framework: str,
    requirements: List[str],
    coverage_pct: float,
    gaps: List[str],
) -> str:
    """
    Hash framework requirement mapping (Hash Function 14).

    Args:
        framework: Regulatory framework identifier.
        requirements: List of requirement IDs mapped.
        coverage_pct: Coverage percentage.
        gaps: List of gap descriptions.

    Returns:
        SHA-256 hash.
    """
    data = {
        "framework": framework,
        "requirements": sorted(requirements),
        "coverage_pct": str(coverage_pct),
        "gaps": sorted(gaps),
    }
    return _compute_hash(data)


def hash_recalculation(
    calculation_id: str,
    trigger_event_id: str,
    trigger_type: str,
    affected_scopes: List[str],
) -> str:
    """
    Hash recalculation trigger (Hash Function 15).

    Args:
        calculation_id: Calculation requiring recalculation.
        trigger_event_id: Event that triggered recalculation.
        trigger_type: Type of trigger.
        affected_scopes: Scopes affected by recalculation.

    Returns:
        SHA-256 hash.
    """
    data = {
        "calculation_id": calculation_id,
        "trigger_event_id": trigger_event_id,
        "trigger_type": trigger_type,
        "affected_scopes": sorted(affected_scopes),
    }
    return _compute_hash(data)


def build_chain_hash(stage_hashes: List[str]) -> str:
    """
    Build final chain hash from all stage hashes (Hash Function 16).

    Chain hash = SHA-256(P1 || P2 || ... || Pn)

    Args:
        stage_hashes: Ordered list of per-stage hashes.

    Returns:
        Final SHA-256 chain hash.
    """
    return _compute_chain_hash(stage_hashes)


# ============================================================================
# SINGLETON ACCESS
# ============================================================================

_tracker_instance: Optional[ProvenanceTracker] = None
_tracker_lock = threading.RLock()


def get_provenance_tracker() -> ProvenanceTracker:
    """
    Get singleton ProvenanceTracker instance.

    Thread-safe singleton pattern.

    Returns:
        ProvenanceTracker instance.

    Example:
        >>> from greenlang.audit_trail_lineage.provenance import (
        ...     get_provenance_tracker,
        ... )
        >>> tracker = get_provenance_tracker()
        >>> chain_id = tracker.start_chain()
    """
    global _tracker_instance
    with _tracker_lock:
        if _tracker_instance is None:
            _tracker_instance = ProvenanceTracker()
        return _tracker_instance


def reset_provenance_tracker() -> None:
    """
    Reset singleton ProvenanceTracker instance.

    Useful for testing. Creates a fresh instance on next access.
    """
    global _tracker_instance
    with _tracker_lock:
        _tracker_instance = None
    ProvenanceTracker.reset_singleton()


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "ProvenanceStage",
    # Models
    "ProvenanceEntry",
    "ProvenanceChain",
    # Tracker
    "ProvenanceTracker",
    # Singleton
    "get_provenance_tracker",
    "reset_provenance_tracker",
    # Hash functions (16)
    "hash_audit_event",
    "hash_lineage_node",
    "hash_lineage_edge",
    "hash_evidence_package",
    "hash_compliance_trace",
    "hash_change_event",
    "hash_chain_verification",
    "hash_pipeline_result",
    "hash_classification",
    "hash_seal",
    "hash_batch_result",
    "hash_stage_result",
    "hash_event_payload",
    "hash_framework_mapping",
    "hash_recalculation",
    "build_chain_hash",
    # Hash utilities
    "_serialize",
    "_compute_hash",
    "_compute_stage_hash",
    "_compute_chain_hash",
]
