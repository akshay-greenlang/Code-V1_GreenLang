# -*- coding: utf-8 -*-
"""
Franchises Provenance Tracking - AGENT-MRV-027

SHA-256 chain-hashed provenance for zero-hallucination audit trails.
10 pipeline stages tracked with deterministic hashing.
Agent: GL-MRV-S3-014

This module implements a complete provenance tracking system for franchise
emissions calculations.  Every step in the calculation pipeline is recorded
with SHA-256 hashes, creating an immutable audit trail that proves no data
was hallucinated or modified.

Pipeline Stages:
    1. VALIDATE - Input validation and boundary checks
    2. CLASSIFY - Franchise type/zone classification
    3. NORMALIZE - Unit and energy normalization
    4. RESOLVE_EFS - Emission factor resolution
    5. CALCULATE - Emissions calculation (zero-hallucination)
    6. ALLOCATE - JV allocation and pro-rata adjustments
    7. AGGREGATE - Network-level aggregation
    8. COMPLIANCE - Regulatory compliance checking
    9. PROVENANCE - Provenance hash chain computation
    10. SEAL - Final chain sealing and output signing

Example:
    >>> tracker = get_provenance_tracker()
    >>> chain_id = tracker.start_chain()
    >>> tracker.record_stage(chain_id, "VALIDATE", input_data, validated_data)
    >>> tracker.record_stage(chain_id, "CALCULATE", calc_input, calc_output)
    >>> final_hash = tracker.seal_chain(chain_id)
    >>> assert tracker.validate_chain(chain_id)

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-014
"""

import hashlib
import json
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

# ============================================================================
# CONSTANTS
# ============================================================================

AGENT_ID = "GL-MRV-S3-014"
AGENT_VERSION = "1.0.0"
HASH_ALGORITHM = "sha256"
ENCODING = "utf-8"


class ProvenanceStage(str, Enum):
    """Pipeline stages for provenance tracking."""

    VALIDATE = "VALIDATE"
    CLASSIFY = "CLASSIFY"
    NORMALIZE = "NORMALIZE"
    RESOLVE_EFS = "RESOLVE_EFS"
    CALCULATE = "CALCULATE"
    ALLOCATE = "ALLOCATE"
    AGGREGATE = "AGGREGATE"
    COMPLIANCE = "COMPLIANCE"
    PROVENANCE = "PROVENANCE"
    SEAL = "SEAL"


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass(frozen=True)
class ProvenanceEntry:
    """
    Single provenance entry in the chain.

    This is a frozen (immutable) record of one stage in the calculation
    pipeline.  The chain_hash links this entry to the previous entry,
    creating an immutable chain.

    Attributes:
        entry_id: Unique identifier for this entry (UUID)
        stage: Pipeline stage name
        timestamp: ISO 8601 UTC timestamp
        input_hash: SHA-256 hash of input data
        output_hash: SHA-256 hash of output data
        chain_hash: SHA-256 hash linking to previous entry
        previous_hash: Chain hash of previous entry (or empty for first)
        agent_id: Agent identifier (GL-MRV-S3-014)
        agent_version: Agent version (1.0.0)
        metadata: Additional context (optional)
    """

    entry_id: str
    stage: str
    timestamp: str
    input_hash: str
    output_hash: str
    chain_hash: str
    previous_hash: str
    agent_id: str = AGENT_ID
    agent_version: str = AGENT_VERSION
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
    Complete provenance chain for a calculation.

    Tracks all entries in order, with validation status and final hash
    when sealed.

    Attributes:
        chain_id: Unique identifier for this chain
        entries: Ordered list of provenance entries
        started_at: ISO 8601 UTC timestamp of chain start
        sealed_at: ISO 8601 UTC timestamp when sealed (or None)
        final_hash: Final SHA-256 hash when sealed (or None)
        chain_valid: Whether chain validation passed
    """

    chain_id: str
    entries: List[ProvenanceEntry] = field(default_factory=list)
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    sealed_at: Optional[str] = None
    final_hash: Optional[str] = None
    chain_valid: bool = True

    @property
    def is_valid(self) -> bool:
        """Whether the chain is valid."""
        return self.chain_valid

    @property
    def root_hash(self) -> str:
        """Root hash of the chain (first entry's chain_hash or empty)."""
        if self.entries:
            return self.entries[0].chain_hash
        return ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert chain to dictionary."""
        return {
            "chain_id": self.chain_id,
            "entries": [e.to_dict() for e in self.entries],
            "started_at": self.started_at,
            "sealed_at": self.sealed_at,
            "final_hash": self.final_hash,
            "chain_valid": self.chain_valid,
            "is_valid": self.is_valid,
            "root_hash": self.root_hash,
        }

    def to_json(self) -> str:
        """Convert chain to JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


@dataclass
class BatchProvenance:
    """
    Provenance tracking for batch calculations.

    Tracks multiple individual chains plus batch-level aggregation.

    Attributes:
        batch_id: Unique identifier for this batch
        individual_chain_ids: List of chain IDs in this batch
        batch_started_at: ISO 8601 UTC timestamp
        batch_sealed_at: ISO 8601 UTC timestamp when sealed (or None)
        batch_hash: Merkle-style aggregate hash (or None)
        item_count: Number of items in batch
    """

    batch_id: str
    individual_chain_ids: List[str] = field(default_factory=list)
    batch_started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    batch_sealed_at: Optional[str] = None
    batch_hash: Optional[str] = None
    item_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert batch provenance to dictionary."""
        return {
            "batch_id": self.batch_id,
            "individual_chain_ids": self.individual_chain_ids,
            "batch_started_at": self.batch_started_at,
            "batch_sealed_at": self.batch_sealed_at,
            "batch_hash": self.batch_hash,
            "item_count": self.item_count,
        }


# ============================================================================
# HASH UTILITIES
# ============================================================================


def _serialize(obj: Any) -> str:
    """
    Serialize object to deterministic JSON string.

    Converts Decimal to string, datetime to ISO format, Enum to value,
    sorts keys, and handles nested dicts/lists and frozen models.

    Args:
        obj: Object to serialize

    Returns:
        Deterministic JSON string
    """

    def default_handler(o: Any) -> Any:
        """Handle non-JSON-serializable types."""
        if isinstance(o, Decimal):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, Enum):
            return o.value
        if hasattr(o, "to_dict"):
            return o.to_dict()
        if hasattr(o, "__dict__"):
            return o.__dict__
        return str(o)

    return json.dumps(obj, sort_keys=True, default=default_handler)


def _compute_hash(data: Any) -> str:
    """
    Compute SHA-256 hash of data.

    Args:
        data: Data to hash (any JSON-serializable type)

    Returns:
        Lowercase hex SHA-256 hash (64 characters)
    """
    serialized = _serialize(data)
    return hashlib.sha256(serialized.encode(ENCODING)).hexdigest()


def _compute_chain_hash(
    previous_hash: str, stage: str, input_hash: str, output_hash: str
) -> str:
    """
    Compute chain hash linking to previous entry.

    Chain hash = SHA-256(previous_hash + stage + input_hash + output_hash)

    Args:
        previous_hash: Chain hash of previous entry (empty string for first)
        stage: Current stage name
        input_hash: Hash of input data
        output_hash: Hash of output data

    Returns:
        SHA-256 chain hash (64 characters)
    """
    chain_data = f"{previous_hash}|{stage}|{input_hash}|{output_hash}"
    return hashlib.sha256(chain_data.encode(ENCODING)).hexdigest()


def _merkle_hash(hashes: List[str]) -> str:
    """
    Compute Merkle-style hash of list of hashes.

    Used for batch aggregation.  Sorts hashes for determinism before
    combining and hashing.

    Args:
        hashes: List of SHA-256 hashes

    Returns:
        Aggregate SHA-256 hash (64 characters)
    """
    if not hashes:
        return hashlib.sha256(b"").hexdigest()
    if len(hashes) == 1:
        return hashes[0]

    # Sort for determinism
    sorted_hashes = sorted(hashes)
    combined = "|".join(sorted_hashes)
    return hashlib.sha256(combined.encode(ENCODING)).hexdigest()


# ============================================================================
# PROVENANCE TRACKER
# ============================================================================


class ProvenanceTracker:
    """
    Main provenance tracking system for franchise calculations.

    Manages multiple chains, records stages, validates integrity.
    Thread-safe with RLock.

    Example:
        >>> tracker = ProvenanceTracker()
        >>> chain_id = tracker.start_chain()
        >>> tracker.record_stage(chain_id, "VALIDATE", input_data, output_data)
        >>> tracker.seal_chain(chain_id)
        >>> assert tracker.validate_chain(chain_id)
    """

    def __init__(
        self,
        agent_id: str = AGENT_ID,
        agent_version: str = AGENT_VERSION,
    ):
        """
        Initialize provenance tracker.

        Args:
            agent_id: Agent identifier
            agent_version: Agent version
        """
        self.agent_id = agent_id
        self.agent_version = agent_version
        self._chains: Dict[str, ProvenanceChain] = {}
        self._lock = threading.RLock()

    def start_chain(self, chain_id: Optional[str] = None) -> str:
        """
        Start a new provenance chain.

        Args:
            chain_id: Optional chain ID (UUID generated if not provided)

        Returns:
            Chain ID

        Raises:
            ValueError: If chain_id already exists
        """
        with self._lock:
            if chain_id is None:
                chain_id = str(uuid.uuid4())

            if chain_id in self._chains:
                raise ValueError(f"Chain {chain_id} already exists")

            chain = ProvenanceChain(chain_id=chain_id)
            self._chains[chain_id] = chain
            return chain_id

    def record_stage(
        self,
        chain_id: str,
        stage: Union[str, ProvenanceStage],
        input_data: Any,
        output_data: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProvenanceEntry:
        """
        Record a pipeline stage in the provenance chain.

        Args:
            chain_id: Chain identifier
            stage: Pipeline stage name
            input_data: Input data for this stage
            output_data: Output data from this stage
            metadata: Optional metadata

        Returns:
            Created ProvenanceEntry

        Raises:
            ValueError: If chain not found or already sealed
        """
        with self._lock:
            chain = self._chains.get(chain_id)
            if chain is None:
                raise ValueError(f"Chain {chain_id} not found")
            if chain.sealed_at is not None:
                raise ValueError(f"Chain {chain_id} already sealed")

            # Convert enum to string
            stage_str = (
                stage.value if isinstance(stage, ProvenanceStage) else stage
            )

            # Compute hashes
            input_hash = _compute_hash(input_data)
            output_hash = _compute_hash(output_data)

            # Get previous hash
            previous_hash = ""
            if chain.entries:
                previous_hash = chain.entries[-1].chain_hash

            # Compute chain hash
            chain_hash = _compute_chain_hash(
                previous_hash, stage_str, input_hash, output_hash
            )

            # Create entry
            entry = ProvenanceEntry(
                entry_id=str(uuid.uuid4()),
                stage=stage_str,
                timestamp=datetime.now(timezone.utc).isoformat(),
                input_hash=input_hash,
                output_hash=output_hash,
                chain_hash=chain_hash,
                previous_hash=previous_hash,
                agent_id=self.agent_id,
                agent_version=self.agent_version,
                metadata=metadata or {},
            )

            # Add to chain
            chain.entries.append(entry)
            return entry

    def validate_chain(self, chain_id: str) -> bool:
        """
        Validate integrity of provenance chain.

        Checks:
        1. Each entry's chain_hash matches recomputed value
        2. Each entry's previous_hash matches previous entry's chain_hash
        3. No gaps in chain

        Args:
            chain_id: Chain identifier

        Returns:
            True if chain is valid, False otherwise

        Raises:
            ValueError: If chain not found
        """
        with self._lock:
            chain = self._chains.get(chain_id)
            if chain is None:
                raise ValueError(f"Chain {chain_id} not found")

            if not chain.entries:
                return True

            # Validate each entry
            for i, entry in enumerate(chain.entries):
                # Check previous_hash
                if i == 0:
                    if entry.previous_hash != "":
                        chain.chain_valid = False
                        return False
                else:
                    if entry.previous_hash != chain.entries[i - 1].chain_hash:
                        chain.chain_valid = False
                        return False

                # Recompute and verify chain_hash
                expected_hash = _compute_chain_hash(
                    entry.previous_hash,
                    entry.stage,
                    entry.input_hash,
                    entry.output_hash,
                )
                if entry.chain_hash != expected_hash:
                    chain.chain_valid = False
                    return False

            chain.chain_valid = True
            return True

    def get_chain(self, chain_id: str) -> ProvenanceChain:
        """
        Get provenance chain.

        Args:
            chain_id: Chain identifier

        Returns:
            ProvenanceChain

        Raises:
            ValueError: If chain not found
        """
        with self._lock:
            chain = self._chains.get(chain_id)
            if chain is None:
                raise ValueError(f"Chain {chain_id} not found")
            return chain

    def seal_chain(self, chain_id: str) -> str:
        """
        Seal the provenance chain.

        Records final timestamp and hash.  No more entries can be added.

        Args:
            chain_id: Chain identifier

        Returns:
            Final hash (64-character hex string)

        Raises:
            ValueError: If chain not found or already sealed
        """
        with self._lock:
            chain = self._chains.get(chain_id)
            if chain is None:
                raise ValueError(f"Chain {chain_id} not found")
            if chain.sealed_at is not None:
                raise ValueError(f"Chain {chain_id} already sealed")

            # Validate before sealing
            if not self.validate_chain(chain_id):
                raise ValueError(f"Chain {chain_id} failed validation")

            # Compute final hash
            if chain.entries:
                final_hash = chain.entries[-1].chain_hash
            else:
                final_hash = hashlib.sha256(
                    chain_id.encode(ENCODING)
                ).hexdigest()

            chain.sealed_at = datetime.now(timezone.utc).isoformat()
            chain.final_hash = final_hash
            return final_hash

    def export_chain(self, chain_id: str, format: str = "json") -> str:
        """
        Export provenance chain.

        Args:
            chain_id: Chain identifier
            format: Export format ("json")

        Returns:
            Serialized chain

        Raises:
            ValueError: If chain not found or format unsupported
        """
        with self._lock:
            chain = self._chains.get(chain_id)
            if chain is None:
                raise ValueError(f"Chain {chain_id} not found")

            if format == "json":
                return chain.to_json()
            else:
                raise ValueError(f"Unsupported format: {format}")

    def verify_entry(self, entry: ProvenanceEntry) -> bool:
        """
        Verify a single provenance entry.

        Recomputes chain_hash and checks it matches.

        Args:
            entry: Provenance entry to verify

        Returns:
            True if entry is valid, False otherwise
        """
        expected_hash = _compute_chain_hash(
            entry.previous_hash,
            entry.stage,
            entry.input_hash,
            entry.output_hash,
        )
        return entry.chain_hash == expected_hash

    def get_stage_hash(
        self,
        chain_id: str,
        stage: Union[str, ProvenanceStage],
    ) -> Optional[str]:
        """
        Get output hash for a specific stage.

        Args:
            chain_id: Chain identifier
            stage: Pipeline stage name

        Returns:
            Output hash for stage, or None if not found

        Raises:
            ValueError: If chain not found
        """
        with self._lock:
            chain = self._chains.get(chain_id)
            if chain is None:
                raise ValueError(f"Chain {chain_id} not found")

            stage_str = (
                stage.value if isinstance(stage, ProvenanceStage) else stage
            )
            for entry in chain.entries:
                if entry.stage == stage_str:
                    return entry.output_hash
            return None

    def reset(self) -> None:
        """
        Clear all provenance chains.

        Primarily used for testing.
        """
        with self._lock:
            self._chains.clear()


# ============================================================================
# SINGLETON ACCESS
# ============================================================================

_tracker_instance: Optional[ProvenanceTracker] = None
_tracker_lock = threading.RLock()


def get_provenance_tracker() -> ProvenanceTracker:
    """
    Get the singleton provenance tracker instance.

    Thread-safe lazy initialization.

    Returns:
        ProvenanceTracker singleton instance

    Example:
        >>> tracker = get_provenance_tracker()
        >>> chain_id = tracker.start_chain()
    """
    global _tracker_instance

    if _tracker_instance is None:
        with _tracker_lock:
            if _tracker_instance is None:
                _tracker_instance = ProvenanceTracker()

    return _tracker_instance


def reset_provenance_tracker() -> None:
    """
    Reset the singleton provenance tracker instance.

    Forces the next call to get_provenance_tracker() to create a
    fresh instance.  Primarily used in testing.
    """
    global _tracker_instance

    with _tracker_lock:
        _tracker_instance = None


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Constants
    "AGENT_ID",
    "AGENT_VERSION",
    "HASH_ALGORITHM",
    "ENCODING",
    # Enum
    "ProvenanceStage",
    # Data models
    "ProvenanceEntry",
    "ProvenanceChain",
    "BatchProvenance",
    # Hash utilities
    "_serialize",
    "_compute_hash",
    "_compute_chain_hash",
    "_merkle_hash",
    # Tracker
    "ProvenanceTracker",
    # Singleton access
    "get_provenance_tracker",
    "reset_provenance_tracker",
]
