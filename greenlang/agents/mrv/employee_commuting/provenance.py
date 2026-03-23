# -*- coding: utf-8 -*-
"""
Provenance Tracking - Employee Commuting Agent (AGENT-MRV-020)

SHA-256 chain hashing for complete audit trail of every calculation.
Supports single-entry chain hashing and batch Merkle-tree aggregation.

Zero-Hallucination Guarantees:
    - Every calculation stage produces a deterministic SHA-256 hash
    - Chain hashing links each stage to the previous for tamper detection
    - Merkle tree aggregation for batch provenance verification
    - All hashing uses hashlib.sha256 with canonical JSON serialization

Pipeline Stages:
    1. VALIDATE - Input validation and data quality checks
    2. CLASSIFY - Commute classification (mode, vehicle type, working pattern)
    3. NORMALIZE - Unit normalization (distance, currency, working days)
    4. RESOLVE_EFS - Emission factor resolution (DEFRA, EPA, IEA)
    5. CALCULATE_COMMUTE - Commute transport emissions calculation
    6. CALCULATE_TELEWORK - Telework/remote work emissions calculation
    7. APPLY_ALLOCATION - Allocation of emissions to reporting entities
    8. COMPLIANCE - Regulatory compliance checking
    9. AGGREGATE - Aggregation and summarization
    10. SEAL - Final sealing and verification

Example:
    >>> tracker = get_provenance_tracker()
    >>> chain = tracker.start_chain("tenant-001", {"source": "survey"})
    >>> tracker.record_stage(
    ...     chain.chain_id, ProvenanceStage.VALIDATE,
    ...     input_data, validated_data,
    ...     engine_id="validation-engine", engine_version="1.0.0"
    ... )
    >>> final_hash = tracker.seal_chain(chain.chain_id)
    >>> valid, errors = tracker.validate_chain(chain.chain_id)
    >>> assert valid and not errors

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-020 Employee Commuting (GL-MRV-S3-007)
Status: Production Ready
"""

import hashlib
import json
import math
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

# ============================================================================
# CONSTANTS
# ============================================================================

AGENT_ID = "GL-MRV-S3-007"
AGENT_VERSION = "1.0.0"
HASH_ALGORITHM = "sha256"
ENCODING = "utf-8"
GENESIS_HASH = "0" * 64  # Sentinel for chain start


# ============================================================================
# PROVENANCE STAGE ENUM
# ============================================================================


class ProvenanceStage(str, Enum):
    """
    Pipeline stages for employee commuting provenance tracking.

    Each stage corresponds to a distinct processing step in the
    employee commuting emissions calculation pipeline. The stages
    are ordered to reflect the typical processing flow from raw
    input validation through final sealing.

    Stages:
        VALIDATE: Input validation and data quality checks. Ensures
            all required fields are present, types are correct, and
            values are within acceptable ranges.
        CLASSIFY: Commute classification. Determines transport mode,
            vehicle type, working pattern (full-time/part-time/hybrid),
            and commute frequency for each employee.
        NORMALIZE: Unit normalization. Converts distances to km,
            currencies to reporting currency, working days to annual
            equivalents, and standardizes date formats.
        RESOLVE_EFS: Emission factor resolution. Selects appropriate
            emission factors from DEFRA, EPA, IEA, or other sources
            based on mode, vehicle type, fuel type, and region.
        CALCULATE_COMMUTE: Commute transport emissions calculation.
            Applies distance-based, spend-based, or average-data methods
            to compute transport CO2e for each commute pattern.
        CALCULATE_TELEWORK: Telework emissions calculation. Computes
            home office energy emissions for remote work days, including
            heating, cooling, lighting, and equipment energy use.
        APPLY_ALLOCATION: Allocation of emissions to reporting entities.
            Applies headcount, FTE, office location, or department-based
            allocation methods to distribute emissions.
        COMPLIANCE: Regulatory compliance checking. Validates results
            against GHG Protocol, CSRD, CDP, SBTi, and other frameworks.
        AGGREGATE: Aggregation and summarization. Computes totals by
            mode, department, location, and other dimensions.
        SEAL: Final sealing and verification. Computes Merkle root hash
            and marks the chain as immutable.
    """

    VALIDATE = "VALIDATE"
    CLASSIFY = "CLASSIFY"
    NORMALIZE = "NORMALIZE"
    RESOLVE_EFS = "RESOLVE_EFS"
    CALCULATE_COMMUTE = "CALCULATE_COMMUTE"
    CALCULATE_TELEWORK = "CALCULATE_TELEWORK"
    APPLY_ALLOCATION = "APPLY_ALLOCATION"
    COMPLIANCE = "COMPLIANCE"
    AGGREGATE = "AGGREGATE"
    SEAL = "SEAL"


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass(frozen=True)
class ProvenanceEntry:
    """
    Single provenance entry in the chain.

    This is a frozen (immutable) record of one stage in the calculation
    pipeline. The chain_hash links this entry to the previous entry,
    creating an immutable chain.

    The entry captures:
        - What went in (input_hash)
        - What came out (output_hash)
        - How it links to the previous step (chain_hash via previous_hash)
        - When it happened (timestamp)
        - Which engine produced it (engine_id, engine_version)
        - How long it took (duration_ms)
        - Optional context (metadata)

    Chain Hash Formula:
        chain_hash = SHA-256(previous_chain_hash | stage | input_hash | output_hash)

    Attributes:
        stage: Pipeline stage name (one of ProvenanceStage values)
        timestamp: ISO 8601 UTC timestamp of when this entry was created
        input_hash: SHA-256 hash of input data for this stage
        output_hash: SHA-256 hash of output data from this stage
        chain_hash: SHA-256 hash linking to previous entry, computed as
                    SHA-256(previous_hash|stage|input_hash|output_hash)
        metadata: Additional context (optional). May include calculation
                  method, emission factor source, uncertainty bounds, etc.
        duration_ms: Processing time for this stage in milliseconds
        engine_id: Identifier of the engine that performed this stage
        engine_version: Version of the engine that performed this stage
    """

    stage: ProvenanceStage
    timestamp: str
    input_hash: str
    output_hash: str
    chain_hash: str
    metadata: Dict[str, Any]
    duration_ms: float
    engine_id: str
    engine_version: str

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert entry to dictionary.

        Returns:
            Dictionary representation of this entry with all fields.
            The stage field is converted to its string value.
        """
        result = asdict(self)
        # Ensure stage is serialized as its string value
        if isinstance(result.get("stage"), Enum):
            result["stage"] = result["stage"].value
        return result

    def to_json(self) -> str:
        """
        Convert entry to deterministic JSON string.

        Returns:
            JSON string with sorted keys for deterministic output.
        """
        return _canonical_json(self.to_dict())


@dataclass
class ProvenanceChain:
    """
    Complete provenance chain for a calculation.

    Tracks all entries in order, with validation status and final hash
    when sealed. Once sealed, no more entries can be added.

    The chain is mutable internally to allow appending entries and
    sealing. The ProvenanceTracker manages all mutations under lock.

    Attributes:
        chain_id: Unique identifier for this chain (UUID)
        tenant_id: Tenant identifier for multi-tenancy isolation
        agent_id: Agent identifier (GL-MRV-S3-007 for employee commuting)
        entries: Ordered list of provenance entries in pipeline order
        started_at: ISO 8601 UTC timestamp of chain start
        sealed_at: ISO 8601 UTC timestamp when sealed (or None)
        final_hash: Final SHA-256 hash when sealed (or None)
        is_sealed: Whether the chain has been sealed (no more additions)
    """

    chain_id: str
    tenant_id: str
    agent_id: str = AGENT_ID
    entries: List[ProvenanceEntry] = field(default_factory=list)
    started_at: str = ""
    sealed_at: Optional[str] = None
    final_hash: Optional[str] = None
    is_sealed: bool = False

    @property
    def root_hash(self) -> str:
        """
        Root hash of the chain (first entry's chain_hash or empty).

        Returns:
            Chain hash of the first entry, or empty string if no entries.
        """
        if self.entries:
            return self.entries[0].chain_hash
        return ""

    @property
    def last_hash(self) -> str:
        """
        Last chain hash in the chain.

        Returns:
            Chain hash of the last entry, or empty string if no entries.
        """
        if self.entries:
            return self.entries[-1].chain_hash
        return ""

    @property
    def entry_count(self) -> int:
        """
        Number of entries in the chain.

        Returns:
            Integer count of entries.
        """
        return len(self.entries)

    @property
    def stages_recorded(self) -> List[str]:
        """
        List of stage names recorded in this chain.

        Returns:
            List of stage name strings in order.
        """
        return [
            e.stage.value if isinstance(e.stage, ProvenanceStage) else e.stage
            for e in self.entries
        ]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert chain to dictionary.

        Returns:
            Dictionary representation of the complete chain with all
            entries expanded.
        """
        return {
            "chain_id": self.chain_id,
            "tenant_id": self.tenant_id,
            "agent_id": self.agent_id,
            "entries": [e.to_dict() for e in self.entries],
            "started_at": self.started_at,
            "sealed_at": self.sealed_at,
            "final_hash": self.final_hash,
            "is_sealed": self.is_sealed,
            "entry_count": self.entry_count,
            "root_hash": self.root_hash,
            "last_hash": self.last_hash,
            "stages_recorded": self.stages_recorded,
        }

    def to_json(self) -> str:
        """
        Convert chain to deterministic JSON string.

        Returns:
            JSON string with sorted keys and indentation.
        """
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str)


# ============================================================================
# HASH UTILITIES
# ============================================================================


def _canonical_json(obj: Any) -> str:
    """
    Serialize object to canonical (deterministic) JSON string.

    Converts Decimal to string, datetime to ISO format, Enum to value,
    sorts keys, and handles nested dicts/lists and frozen models. This
    is the foundation of all provenance hashing. Deterministic
    serialization ensures that the same input always produces the same
    hash, regardless of dict ordering or Python object representation.

    Args:
        obj: Object to serialize. Supports dict, list, str, int, float,
             Decimal, datetime, Enum, dataclasses with to_dict(), and
             any object with __dict__.

    Returns:
        Deterministic JSON string with sorted keys.

    Example:
        >>> _canonical_json({"b": 1, "a": 2})
        '{"a": 2, "b": 1}'
        >>> _canonical_json(Decimal("123.456"))
        '"123.456"'
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


def _sha256(data: str) -> str:
    """
    Compute SHA-256 hash of a UTF-8 string.

    This is the lowest-level hashing primitive. All other hash functions
    ultimately call this method.

    Args:
        data: UTF-8 string to hash.

    Returns:
        Lowercase hex SHA-256 hash (64 characters).

    Example:
        >>> _sha256("hello")
        '2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824'
    """
    return hashlib.sha256(data.encode(ENCODING)).hexdigest()


def _serialize(obj: Any) -> str:
    """
    Serialize object to deterministic JSON string.

    Alias for _canonical_json for backward compatibility and convenience.

    Args:
        obj: Object to serialize.

    Returns:
        Deterministic JSON string with sorted keys.
    """
    return _canonical_json(obj)


def _compute_hash(data: Any) -> str:
    """
    Compute SHA-256 hash of data.

    Serializes the data to a deterministic JSON string via _canonical_json,
    then computes the SHA-256 hash of the UTF-8 encoded string.

    Args:
        data: Data to hash (any JSON-serializable type, or any type
              supported by _canonical_json).

    Returns:
        Lowercase hex SHA-256 hash (64 characters).

    Example:
        >>> _compute_hash({"mode": "car", "distance_km": 25.0})
        'a1b2c3...'  # 64-character hex string
    """
    serialized = _canonical_json(data)
    return _sha256(serialized)


def _compute_chain_hash(
    previous_hash: str, stage: str, input_hash: str, output_hash: str
) -> str:
    """
    Compute chain hash linking to previous entry.

    Chain hash = SHA-256(previous_hash|stage|input_hash|output_hash)

    The pipe-delimited concatenation ensures that the chain hash depends
    on all four components, preventing substitution attacks where one
    component is swapped with another.

    Args:
        previous_hash: Chain hash of previous entry (empty string for first)
        stage: Current stage name (e.g., "VALIDATE", "CALCULATE_COMMUTE")
        input_hash: SHA-256 hash of input data
        output_hash: SHA-256 hash of output data

    Returns:
        SHA-256 chain hash (64-character hex string)

    Example:
        >>> _compute_chain_hash("", "VALIDATE", "abc123...", "def456...")
        'f1e2d3...'  # 64-character hex string
    """
    chain_data = f"{previous_hash}|{stage}|{input_hash}|{output_hash}"
    return _sha256(chain_data)


def _merkle_hash(hashes: List[str]) -> str:
    """
    Compute Merkle-tree root hash of a list of hashes.

    Implements a binary Merkle tree. If the number of hashes at any level
    is odd, the last hash is duplicated. Leaf nodes are the input hashes,
    and each parent node is SHA-256(left_child|right_child).

    For batch aggregation and chain sealing. The tree structure provides
    efficient proof-of-inclusion for any individual chain within a batch.

    Args:
        hashes: List of SHA-256 hashes to aggregate.

    Returns:
        Merkle root SHA-256 hash (64-character hex string).

    Example:
        >>> _merkle_hash(["abc123...", "def456..."])
        '789xyz...'  # 64-character hex string
        >>> _merkle_hash([])
        'e3b0c44...'  # SHA-256 of empty bytes
    """
    if not hashes:
        return hashlib.sha256(b"").hexdigest()
    if len(hashes) == 1:
        return hashes[0]

    # Sort for determinism before building tree
    current_level = sorted(hashes)

    while len(current_level) > 1:
        next_level: List[str] = []
        for i in range(0, len(current_level), 2):
            left = current_level[i]
            # Duplicate last if odd
            right = current_level[i + 1] if i + 1 < len(current_level) else left
            combined = f"{left}|{right}"
            next_level.append(_sha256(combined))
        current_level = next_level

    return current_level[0]


def _merkle_proof(hashes: List[str], target_index: int) -> List[Tuple[str, str]]:
    """
    Build a Merkle proof for the hash at target_index.

    The proof is a list of (sibling_hash, direction) tuples where
    direction is 'L' or 'R' indicating which side the sibling is on.

    Args:
        hashes: Sorted list of leaf hashes used to build the tree.
        target_index: Index of the target hash in the sorted list.

    Returns:
        List of (sibling_hash, direction) tuples forming the proof path.

    Raises:
        ValueError: If target_index is out of range.
    """
    if not hashes:
        return []
    if target_index < 0 or target_index >= len(hashes):
        raise ValueError(f"target_index {target_index} out of range [0, {len(hashes)})")

    current_level = list(hashes)
    proof: List[Tuple[str, str]] = []
    idx = target_index

    while len(current_level) > 1:
        next_level: List[str] = []
        for i in range(0, len(current_level), 2):
            left = current_level[i]
            right = current_level[i + 1] if i + 1 < len(current_level) else left

            if i == idx or i + 1 == idx:
                if idx % 2 == 0:
                    # Target is on left, sibling is on right
                    sibling = right
                    proof.append((sibling, "R"))
                else:
                    # Target is on right, sibling is on left
                    sibling = left
                    proof.append((sibling, "L"))

            combined = f"{left}|{right}"
            next_level.append(_sha256(combined))

        idx = idx // 2
        current_level = next_level

    return proof


def _verify_merkle_proof(
    target_hash: str, proof: List[Tuple[str, str]], root_hash: str
) -> bool:
    """
    Verify a Merkle proof against a root hash.

    Args:
        target_hash: The leaf hash to verify.
        proof: List of (sibling_hash, direction) tuples.
        root_hash: Expected Merkle root hash.

    Returns:
        True if the proof is valid, False otherwise.
    """
    current = target_hash
    for sibling_hash, direction in proof:
        if direction == "L":
            combined = f"{sibling_hash}|{current}"
        else:
            combined = f"{current}|{sibling_hash}"
        current = _sha256(combined)
    return current == root_hash


# ============================================================================
# PROVENANCE TRACKER
# ============================================================================


class ProvenanceTracker:
    """
    Main provenance tracking system for employee commuting calculations.

    Manages multiple chains, records stages, validates integrity, and
    exports audit trails. Thread-safe with threading.RLock.

    Each calculation gets its own chain (identified by chain_id). Stages
    are recorded sequentially, with each stage's chain_hash depending on
    the previous stage's chain_hash, creating an immutable linked chain.

    The tracker uses RLock (reentrant lock) to allow methods that call
    other locked methods without deadlocking (e.g., seal_chain calls
    validate_chain internally).

    Example:
        >>> tracker = ProvenanceTracker()
        >>> chain = tracker.start_chain("tenant-001")
        >>> tracker.record_stage(
        ...     chain.chain_id, ProvenanceStage.VALIDATE,
        ...     input_data, output_data,
        ...     engine_id="validation-engine", engine_version="1.0.0"
        ... )
        >>> final_hash = tracker.seal_chain(chain.chain_id)
        >>> valid, errors = tracker.validate_chain(chain.chain_id)
        >>> assert valid
    """

    def __init__(
        self, agent_id: str = AGENT_ID, agent_version: str = AGENT_VERSION
    ):
        """
        Initialize provenance tracker.

        Args:
            agent_id: Agent identifier (default: GL-MRV-S3-007)
            agent_version: Agent version (default: 1.0.0)
        """
        self.agent_id = agent_id
        self.agent_version = agent_version
        self._chains: Dict[str, ProvenanceChain] = {}
        self._lock = threading.RLock()

    def start_chain(
        self,
        tenant_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        chain_id: Optional[str] = None,
    ) -> ProvenanceChain:
        """
        Start a new provenance chain.

        Creates a new empty chain with a unique identifier and associates
        it with the given tenant. The chain starts in an unsealed state,
        ready to accept stage recordings.

        Args:
            tenant_id: Tenant identifier for multi-tenancy isolation.
            metadata: Optional metadata to associate with chain creation.
                      Stored as metadata on an implicit genesis entry if
                      provided. Common fields: source, calculation_type,
                      reporting_year.
            chain_id: Optional chain ID. If not provided, a UUID v4 is
                      generated automatically.

        Returns:
            ProvenanceChain - the newly created chain object.

        Raises:
            ValueError: If chain_id already exists in this tracker.

        Example:
            >>> chain = tracker.start_chain("tenant-001", {"year": 2025})
            >>> assert chain.tenant_id == "tenant-001"
        """
        with self._lock:
            if chain_id is None:
                chain_id = str(uuid.uuid4())

            if chain_id in self._chains:
                raise ValueError(f"Chain {chain_id} already exists")

            now = datetime.now(timezone.utc).isoformat()
            chain = ProvenanceChain(
                chain_id=chain_id,
                tenant_id=tenant_id,
                agent_id=self.agent_id,
                entries=[],
                started_at=now,
                sealed_at=None,
                final_hash=None,
                is_sealed=False,
            )
            self._chains[chain_id] = chain

            # If metadata provided, record it as context on the chain
            # We store creation metadata as a property accessible later
            if metadata:
                chain._creation_metadata = metadata  # type: ignore[attr-defined]

            return chain

    def record_stage(
        self,
        chain_id: str,
        stage: Union[str, ProvenanceStage],
        input_data: Any,
        output_data: Any,
        engine_id: str = "unknown",
        engine_version: str = "0.0.0",
        metadata: Optional[Dict[str, Any]] = None,
        duration_ms: float = 0.0,
    ) -> ProvenanceEntry:
        """
        Record a pipeline stage in the provenance chain.

        Computes SHA-256 hashes of input and output data, links to the
        previous entry via chain_hash, and appends the entry to the chain.

        Args:
            chain_id: Chain identifier.
            stage: Pipeline stage name. Can be a ProvenanceStage enum or
                   a string matching one of the stage names.
            input_data: Input data for this stage. Any JSON-serializable
                        type. Will be hashed but not stored.
            output_data: Output data from this stage. Any JSON-serializable
                         type. Will be hashed but not stored.
            engine_id: Identifier of the engine that performed this stage
                       (e.g., "commute-distance-engine", "telework-engine").
            engine_version: Semantic version of the engine (e.g., "1.0.0").
            metadata: Optional metadata dictionary. May include calculation
                      method, emission factor source, uncertainty bounds,
                      working days, survey response count, etc.
            duration_ms: Processing time for this stage in milliseconds.
                         If 0.0, caller did not track timing.

        Returns:
            Created ProvenanceEntry (frozen, immutable).

        Raises:
            ValueError: If chain not found or already sealed.

        Example:
            >>> entry = tracker.record_stage(
            ...     chain.chain_id,
            ...     ProvenanceStage.CALCULATE_COMMUTE,
            ...     {"distance_km": 25.0, "mode": "car"},
            ...     {"co2e_kg": 1825.0},
            ...     engine_id="commute-distance-engine",
            ...     engine_version="1.0.0",
            ...     metadata={"method": "distance_based"},
            ...     duration_ms=12.5
            ... )
        """
        with self._lock:
            chain = self._chains.get(chain_id)
            if chain is None:
                raise ValueError(f"Chain {chain_id} not found")
            if chain.is_sealed:
                raise ValueError(f"Chain {chain_id} already sealed")

            # Convert string to enum if needed
            if isinstance(stage, str):
                try:
                    stage_enum = ProvenanceStage(stage)
                except ValueError:
                    stage_enum = ProvenanceStage(stage)
            else:
                stage_enum = stage

            stage_str = stage_enum.value

            # Compute hashes of input and output data
            input_hash = _compute_hash(input_data)
            output_hash = _compute_hash(output_data)

            # Get previous chain hash
            previous_chain_hash = ""
            if chain.entries:
                previous_chain_hash = chain.entries[-1].chain_hash

            # Compute chain hash linking to previous entry
            chain_hash = _compute_chain_hash(
                previous_chain_hash, stage_str, input_hash, output_hash
            )

            # Create frozen entry
            entry = ProvenanceEntry(
                stage=stage_enum,
                timestamp=datetime.now(timezone.utc).isoformat(),
                input_hash=input_hash,
                output_hash=output_hash,
                chain_hash=chain_hash,
                metadata=metadata if metadata is not None else {},
                duration_ms=duration_ms,
                engine_id=engine_id,
                engine_version=engine_version,
            )

            # Append to chain
            chain.entries.append(entry)
            return entry

    def validate_chain(self, chain_id: str) -> Tuple[bool, List[str]]:
        """
        Validate integrity of provenance chain.

        Performs comprehensive integrity checks on every entry:
            1. First entry's previous chain hash must be empty string
            2. Each subsequent entry's previous hash must match the
               chain_hash of the preceding entry
            3. Each entry's chain_hash must match the recomputed value
               from (previous_hash, stage, input_hash, output_hash)
            4. Timestamps must be monotonically non-decreasing
            5. All entries must have valid stage names

        Args:
            chain_id: Chain identifier.

        Returns:
            Tuple of (is_valid, error_messages).
            - is_valid: True if chain passes all checks.
            - error_messages: List of error strings (empty if valid).

        Raises:
            ValueError: If chain not found.

        Example:
            >>> valid, errors = tracker.validate_chain(chain_id)
            >>> if not valid:
            ...     for err in errors:
            ...         print(f"INTEGRITY ERROR: {err}")
        """
        with self._lock:
            chain = self._chains.get(chain_id)
            if chain is None:
                raise ValueError(f"Chain {chain_id} not found")

            errors: List[str] = []
            entries = chain.entries

            if not entries:
                return True, []

            # Check each entry in sequence
            for i, entry in enumerate(entries):
                # Check 1: Previous hash linkage
                if i == 0:
                    if entry.chain_hash != _compute_chain_hash(
                        "", self._stage_value(entry.stage),
                        entry.input_hash, entry.output_hash
                    ):
                        errors.append(
                            f"Entry {i} ({self._stage_value(entry.stage)}): "
                            f"chain_hash mismatch for genesis entry"
                        )
                else:
                    expected_prev = entries[i - 1].chain_hash
                    # Recompute chain hash using previous entry's chain_hash
                    expected_chain_hash = _compute_chain_hash(
                        expected_prev,
                        self._stage_value(entry.stage),
                        entry.input_hash,
                        entry.output_hash,
                    )
                    if entry.chain_hash != expected_chain_hash:
                        errors.append(
                            f"Entry {i} ({self._stage_value(entry.stage)}): "
                            f"chain_hash mismatch. Expected {expected_chain_hash[:16]}..., "
                            f"got {entry.chain_hash[:16]}..."
                        )

                # Check 2: Validate stage is a known stage
                stage_val = self._stage_value(entry.stage)
                valid_stages = {s.value for s in ProvenanceStage}
                if stage_val not in valid_stages:
                    errors.append(
                        f"Entry {i}: unknown stage '{stage_val}'"
                    )

                # Check 3: Timestamp monotonicity
                if i > 0:
                    prev_ts = entries[i - 1].timestamp
                    curr_ts = entry.timestamp
                    if curr_ts < prev_ts:
                        errors.append(
                            f"Entry {i} ({self._stage_value(entry.stage)}): "
                            f"timestamp {curr_ts} is before previous {prev_ts}"
                        )

                # Check 4: Hash format validation (64 hex chars)
                for hash_name, hash_val in [
                    ("input_hash", entry.input_hash),
                    ("output_hash", entry.output_hash),
                    ("chain_hash", entry.chain_hash),
                ]:
                    if len(hash_val) != 64:
                        errors.append(
                            f"Entry {i} ({self._stage_value(entry.stage)}): "
                            f"{hash_name} has invalid length {len(hash_val)}, expected 64"
                        )
                    elif not all(c in "0123456789abcdef" for c in hash_val):
                        errors.append(
                            f"Entry {i} ({self._stage_value(entry.stage)}): "
                            f"{hash_name} contains non-hex characters"
                        )

            is_valid = len(errors) == 0
            return is_valid, errors

    def seal_chain(self, chain_id: str) -> str:
        """
        Seal the provenance chain.

        Records final timestamp and computes the Merkle root hash from
        all entry chain_hashes. Once sealed, no more entries can be added
        to this chain.

        The sealing process:
            1. Validates chain integrity
            2. Collects all entry chain_hashes
            3. Computes Merkle-tree aggregate hash
            4. Records sealed timestamp
            5. Marks chain as sealed
            6. Returns final hash

        Args:
            chain_id: Chain identifier.

        Returns:
            Final Merkle root hash (64-character hex string).

        Raises:
            ValueError: If chain not found, already sealed, or fails validation.

        Example:
            >>> final_hash = tracker.seal_chain(chain_id)
            >>> assert len(final_hash) == 64
        """
        with self._lock:
            chain = self._chains.get(chain_id)
            if chain is None:
                raise ValueError(f"Chain {chain_id} not found")
            if chain.is_sealed:
                raise ValueError(f"Chain {chain_id} already sealed")

            # Validate before sealing
            is_valid, validation_errors = self.validate_chain(chain_id)
            if not is_valid:
                error_summary = "; ".join(validation_errors[:5])
                raise ValueError(
                    f"Chain {chain_id} failed validation: {error_summary}"
                )

            # Compute Merkle root from all entry chain_hashes
            if chain.entries:
                all_hashes = [e.chain_hash for e in chain.entries]
                final_hash = _merkle_hash(all_hashes)
            else:
                final_hash = hashlib.sha256(chain_id.encode(ENCODING)).hexdigest()

            chain.sealed_at = datetime.now(timezone.utc).isoformat()
            chain.final_hash = final_hash
            chain.is_sealed = True
            return final_hash

    def get_chain(self, chain_id: str) -> Optional[ProvenanceChain]:
        """
        Get provenance chain.

        Returns the chain object if found, or None if not found. Unlike
        other methods that raise ValueError, this returns None for
        missing chains to support safe lookups.

        Args:
            chain_id: Chain identifier.

        Returns:
            ProvenanceChain if found, None otherwise.

        Example:
            >>> chain = tracker.get_chain(chain_id)
            >>> if chain is not None:
            ...     print(f"Chain has {chain.entry_count} entries")
        """
        with self._lock:
            return self._chains.get(chain_id)

    def export_chain(self, chain_id: str) -> Dict[str, Any]:
        """
        Export provenance chain as a dictionary.

        Returns the full chain data including all entries, suitable for
        JSON serialization, database storage, or API responses.

        Args:
            chain_id: Chain identifier.

        Returns:
            Dictionary representation of the complete chain.

        Raises:
            ValueError: If chain not found.

        Example:
            >>> chain_dict = tracker.export_chain(chain_id)
            >>> json_str = json.dumps(chain_dict, indent=2)
        """
        with self._lock:
            chain = self._chains.get(chain_id)
            if chain is None:
                raise ValueError(f"Chain {chain_id} not found")
            return chain.to_dict()

    def verify_entry(
        self, entry: ProvenanceEntry, previous_chain_hash: str = ""
    ) -> bool:
        """
        Verify a single provenance entry.

        Recomputes chain_hash from the entry's components and the given
        previous_chain_hash, then checks that it matches the stored
        chain_hash.

        Args:
            entry: Provenance entry to verify.
            previous_chain_hash: Chain hash of the previous entry in the
                                 chain. Empty string for the first entry.

        Returns:
            True if entry is valid, False otherwise.

        Example:
            >>> entry = tracker.record_stage(chain_id, "VALIDATE", inp, out)
            >>> assert tracker.verify_entry(entry, "")
        """
        stage_str = self._stage_value(entry.stage)
        expected_hash = _compute_chain_hash(
            previous_chain_hash,
            stage_str,
            entry.input_hash,
            entry.output_hash,
        )
        return entry.chain_hash == expected_hash

    def delete_chain(self, chain_id: str) -> bool:
        """
        Delete a provenance chain.

        Permanently removes the chain from the tracker. This operation
        cannot be undone.

        Args:
            chain_id: Chain identifier.

        Returns:
            True if deleted, False if not found.

        Example:
            >>> deleted = tracker.delete_chain(chain_id)
            >>> assert deleted
        """
        with self._lock:
            if chain_id in self._chains:
                del self._chains[chain_id]
                return True
            return False

    def list_chains(
        self,
        tenant_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[ProvenanceChain]:
        """
        List provenance chains with optional tenant filtering and pagination.

        Args:
            tenant_id: Optional tenant ID to filter by. If None, returns
                       chains for all tenants.
            limit: Maximum number of chains to return (default 100).
            offset: Number of chains to skip (default 0).

        Returns:
            List of ProvenanceChain objects matching the criteria.

        Example:
            >>> chains = tracker.list_chains("tenant-001", limit=10)
            >>> for chain in chains:
            ...     print(chain.chain_id, chain.is_sealed)
        """
        with self._lock:
            all_chains = list(self._chains.values())

            # Filter by tenant if specified
            if tenant_id is not None:
                all_chains = [c for c in all_chains if c.tenant_id == tenant_id]

            # Sort by started_at for deterministic ordering
            all_chains.sort(key=lambda c: c.started_at)

            # Apply pagination
            return all_chains[offset: offset + limit]

    def get_chain_summary(self, chain_id: str) -> Dict[str, Any]:
        """
        Get summary of provenance chain.

        Returns a lightweight summary without full entry details,
        suitable for logging, monitoring, and dashboard display.

        Args:
            chain_id: Chain identifier.

        Returns:
            Summary dictionary with key metrics including chain_id,
            tenant_id, agent info, timestamps, entry count, stages,
            sealing status, and hash information.

        Raises:
            ValueError: If chain not found.

        Example:
            >>> summary = tracker.get_chain_summary(chain_id)
            >>> print(summary['entry_count'])
            10
        """
        with self._lock:
            chain = self._chains.get(chain_id)
            if chain is None:
                raise ValueError(f"Chain {chain_id} not found")

            return {
                "chain_id": chain.chain_id,
                "tenant_id": chain.tenant_id,
                "agent_id": chain.agent_id,
                "started_at": chain.started_at,
                "sealed_at": chain.sealed_at,
                "final_hash": chain.final_hash,
                "is_sealed": chain.is_sealed,
                "entry_count": chain.entry_count,
                "stages": chain.stages_recorded,
                "root_hash": chain.root_hash,
                "last_hash": chain.last_hash,
                "duration_total_ms": sum(e.duration_ms for e in chain.entries),
                "engines_used": list({e.engine_id for e in chain.entries}),
            }

    def get_stage_hash(
        self, chain_id: str, stage: Union[str, ProvenanceStage]
    ) -> Optional[str]:
        """
        Get output hash for a specific stage.

        Searches the chain for the first entry matching the given stage
        and returns its output_hash.

        Args:
            chain_id: Chain identifier.
            stage: Pipeline stage name (string or ProvenanceStage enum).

        Returns:
            Output hash for stage, or None if stage not found in chain.

        Raises:
            ValueError: If chain not found.

        Example:
            >>> validate_hash = tracker.get_stage_hash(chain_id, "VALIDATE")
        """
        with self._lock:
            chain = self._chains.get(chain_id)
            if chain is None:
                raise ValueError(f"Chain {chain_id} not found")

            target = stage.value if isinstance(stage, ProvenanceStage) else stage
            for entry in chain.entries:
                if self._stage_value(entry.stage) == target:
                    return entry.output_hash
            return None

    def reset(self) -> None:
        """
        Clear all chains tracked by this tracker instance.

        Useful for testing. Removes all chain state.

        Example:
            >>> tracker.reset()
            >>> assert len(tracker.get_all_chains()) == 0
        """
        with self._lock:
            self._chains.clear()

    def get_all_chains(self) -> List[str]:
        """
        Get list of all chain IDs.

        Returns:
            List of chain IDs managed by this tracker.

        Example:
            >>> chain_ids = tracker.get_all_chains()
        """
        with self._lock:
            return list(self._chains.keys())

    def clear_all_chains(self) -> int:
        """
        Clear all chains (for testing).

        Returns:
            Number of chains cleared.
        """
        with self._lock:
            count = len(self._chains)
            self._chains.clear()
            return count

    @staticmethod
    def _stage_value(stage: Union[str, ProvenanceStage]) -> str:
        """
        Extract string value from stage (enum or str).

        Args:
            stage: ProvenanceStage enum or string.

        Returns:
            String value of the stage.
        """
        if isinstance(stage, ProvenanceStage):
            return stage.value
        return str(stage)


# ============================================================================
# BATCH PROVENANCE TRACKER
# ============================================================================


@dataclass
class BatchProvenance:
    """
    Provenance tracking record for batch calculations.

    Tracks multiple individual chains plus batch-level Merkle tree
    aggregation. Used when processing an entire employee population's
    commuting data in one batch job.

    Attributes:
        batch_id: Unique identifier for this batch (UUID).
        tenant_id: Tenant identifier for multi-tenancy isolation.
        individual_chain_ids: List of chain IDs in this batch.
        batch_started_at: ISO 8601 UTC timestamp of batch start.
        batch_sealed_at: ISO 8601 UTC timestamp when sealed (or None).
        batch_hash: Merkle-tree root hash (or None until sealed).
        batch_size: Expected number of items in batch.
        merkle_tree_levels: Number of levels in the Merkle tree (set on seal).
        merkle_leaf_hashes: Sorted leaf hashes used to build the tree.
    """

    batch_id: str
    tenant_id: str
    individual_chain_ids: List[str] = field(default_factory=list)
    batch_started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    batch_sealed_at: Optional[str] = None
    batch_hash: Optional[str] = None
    batch_size: int = 0
    merkle_tree_levels: int = 0
    merkle_leaf_hashes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert batch provenance to dictionary.

        Returns:
            Dictionary representation of the batch provenance.
        """
        return {
            "batch_id": self.batch_id,
            "tenant_id": self.tenant_id,
            "individual_chain_ids": list(self.individual_chain_ids),
            "batch_started_at": self.batch_started_at,
            "batch_sealed_at": self.batch_sealed_at,
            "batch_hash": self.batch_hash,
            "batch_size": self.batch_size,
            "chain_count": len(self.individual_chain_ids),
            "merkle_tree_levels": self.merkle_tree_levels,
            "is_sealed": self.batch_sealed_at is not None,
        }


class BatchProvenanceTracker:
    """
    Provenance tracking for batch employee commuting calculations.

    Manages multiple individual chains plus batch-level Merkle-tree
    aggregation. Each batch contains references to individual employee
    calculation chains, and produces a Merkle root hash on sealing.

    Supports Merkle proof generation and verification for efficient
    proof-of-inclusion of any individual chain within a batch.

    Thread-safe with threading.RLock.

    Example:
        >>> batch_tracker = BatchProvenanceTracker(tracker)
        >>> batch_id = batch_tracker.start_batch("tenant-001", 5000)
        >>> for employee in employees:
        ...     chain = tracker.start_chain("tenant-001")
        ...     # ... record stages for this employee ...
        ...     tracker.seal_chain(chain.chain_id)
        ...     batch_tracker.add_chain(batch_id, chain)
        >>> root_hash = batch_tracker.build_merkle_tree(batch_id)
        >>> assert batch_tracker.verify_merkle_proof(batch_id, chain.chain_id)
    """

    def __init__(self, tracker: ProvenanceTracker):
        """
        Initialize batch provenance tracker.

        Args:
            tracker: ProvenanceTracker instance for individual chains.
                     Used to retrieve chain final hashes during batch
                     sealing and Merkle tree construction.
        """
        self.tracker = tracker
        self._batches: Dict[str, BatchProvenance] = {}
        self._lock = threading.RLock()

    def start_batch(
        self,
        tenant_id: str,
        batch_size: int = 0,
        batch_id: Optional[str] = None,
    ) -> str:
        """
        Start a new batch.

        Args:
            tenant_id: Tenant identifier for multi-tenancy isolation.
            batch_size: Expected number of items in the batch. Used for
                        progress tracking and validation.
            batch_id: Optional batch ID. If not provided, a UUID v4 is
                      generated automatically.

        Returns:
            Batch ID (str).

        Raises:
            ValueError: If batch_id already exists.

        Example:
            >>> batch_id = batch_tracker.start_batch("tenant-001", 5000)
        """
        with self._lock:
            if batch_id is None:
                batch_id = str(uuid.uuid4())

            if batch_id in self._batches:
                raise ValueError(f"Batch {batch_id} already exists")

            batch = BatchProvenance(
                batch_id=batch_id,
                tenant_id=tenant_id,
                individual_chain_ids=[],
                batch_started_at=datetime.now(timezone.utc).isoformat(),
                batch_sealed_at=None,
                batch_hash=None,
                batch_size=batch_size,
                merkle_tree_levels=0,
                merkle_leaf_hashes=[],
            )
            self._batches[batch_id] = batch
            return batch_id

    def add_chain(
        self, batch_id: str, chain: Union[ProvenanceChain, str]
    ) -> None:
        """
        Add a chain to the batch.

        Accepts either a ProvenanceChain object or a chain_id string.

        Args:
            batch_id: Batch identifier.
            chain: ProvenanceChain object or chain_id string to add.

        Raises:
            ValueError: If batch not found or already sealed.

        Example:
            >>> batch_tracker.add_chain(batch_id, chain)
            >>> batch_tracker.add_chain(batch_id, "chain-uuid-string")
        """
        with self._lock:
            batch = self._batches.get(batch_id)
            if batch is None:
                raise ValueError(f"Batch {batch_id} not found")
            if batch.batch_sealed_at is not None:
                raise ValueError(f"Batch {batch_id} already sealed")

            if isinstance(chain, ProvenanceChain):
                chain_id = chain.chain_id
            else:
                chain_id = chain

            batch.individual_chain_ids.append(chain_id)

    def build_merkle_tree(self, batch_id: str) -> str:
        """
        Build Merkle tree from all chain final hashes and seal the batch.

        Collects the final_hash from each chain in the batch, builds a
        binary Merkle tree, and stores the root hash. Chains that are not
        sealed or not found are skipped (their hashes are not included).

        The sorted leaf hashes are stored on the batch for later
        proof generation and verification.

        Args:
            batch_id: Batch identifier.

        Returns:
            Merkle root hash (64-character hex string).

        Raises:
            ValueError: If batch not found or already sealed.

        Example:
            >>> root_hash = batch_tracker.build_merkle_tree(batch_id)
            >>> assert len(root_hash) == 64
        """
        with self._lock:
            batch = self._batches.get(batch_id)
            if batch is None:
                raise ValueError(f"Batch {batch_id} not found")
            if batch.batch_sealed_at is not None:
                raise ValueError(f"Batch {batch_id} already sealed")

            # Collect final hashes from all sealed chains
            chain_hashes: List[str] = []
            for chain_id in batch.individual_chain_ids:
                chain = self.tracker.get_chain(chain_id)
                if chain is not None and chain.final_hash is not None:
                    chain_hashes.append(chain.final_hash)

            # Sort for determinism and store leaf hashes
            sorted_hashes = sorted(chain_hashes)
            batch.merkle_leaf_hashes = sorted_hashes

            # Compute tree depth
            if sorted_hashes:
                batch.merkle_tree_levels = math.ceil(math.log2(len(sorted_hashes))) + 1
            else:
                batch.merkle_tree_levels = 0

            # Build Merkle root
            root_hash = _merkle_hash(sorted_hashes)
            batch.batch_hash = root_hash
            batch.batch_sealed_at = datetime.now(timezone.utc).isoformat()
            return root_hash

    def verify_merkle_proof(self, batch_id: str, chain_id: str) -> bool:
        """
        Verify that a chain is included in the batch Merkle tree.

        Generates a Merkle proof for the given chain and verifies it
        against the batch root hash.

        Args:
            batch_id: Batch identifier.
            chain_id: Chain identifier to verify inclusion of.

        Returns:
            True if the chain is provably included in the batch,
            False otherwise.

        Raises:
            ValueError: If batch not found or batch not sealed.

        Example:
            >>> included = batch_tracker.verify_merkle_proof(batch_id, chain_id)
            >>> assert included, "Chain not in batch"
        """
        with self._lock:
            batch = self._batches.get(batch_id)
            if batch is None:
                raise ValueError(f"Batch {batch_id} not found")
            if batch.batch_sealed_at is None:
                raise ValueError(f"Batch {batch_id} not yet sealed")
            if batch.batch_hash is None:
                return False

            # Get chain's final hash
            chain = self.tracker.get_chain(chain_id)
            if chain is None or chain.final_hash is None:
                return False

            target_hash = chain.final_hash
            leaf_hashes = batch.merkle_leaf_hashes

            # Find target in sorted leaf hashes
            if target_hash not in leaf_hashes:
                return False

            target_index = leaf_hashes.index(target_hash)

            # Build proof and verify
            proof = _merkle_proof(leaf_hashes, target_index)
            return _verify_merkle_proof(target_hash, proof, batch.batch_hash)

    def get_batch_summary(self, batch_id: str) -> Dict[str, Any]:
        """
        Get summary of batch provenance.

        Args:
            batch_id: Batch identifier.

        Returns:
            Summary dictionary with batch metadata, chain count,
            Merkle tree info, and sealing status.

        Raises:
            ValueError: If batch not found.

        Example:
            >>> summary = batch_tracker.get_batch_summary(batch_id)
            >>> print(summary['chain_count'])
        """
        with self._lock:
            batch = self._batches.get(batch_id)
            if batch is None:
                raise ValueError(f"Batch {batch_id} not found")

            return {
                "batch_id": batch.batch_id,
                "tenant_id": batch.tenant_id,
                "batch_started_at": batch.batch_started_at,
                "batch_sealed_at": batch.batch_sealed_at,
                "batch_hash": batch.batch_hash,
                "batch_size": batch.batch_size,
                "chain_count": len(batch.individual_chain_ids),
                "individual_chain_ids": list(batch.individual_chain_ids),
                "is_sealed": batch.batch_sealed_at is not None,
                "merkle_tree_levels": batch.merkle_tree_levels,
                "merkle_leaf_count": len(batch.merkle_leaf_hashes),
            }

    def get_batch(self, batch_id: str) -> BatchProvenance:
        """
        Get batch provenance object.

        Args:
            batch_id: Batch identifier.

        Returns:
            BatchProvenance object.

        Raises:
            ValueError: If batch not found.

        Example:
            >>> batch = batch_tracker.get_batch(batch_id)
        """
        with self._lock:
            batch = self._batches.get(batch_id)
            if batch is None:
                raise ValueError(f"Batch {batch_id} not found")
            return batch

    def delete_batch(self, batch_id: str) -> bool:
        """
        Delete a batch provenance record.

        Args:
            batch_id: Batch identifier.

        Returns:
            True if deleted, False if not found.
        """
        with self._lock:
            if batch_id in self._batches:
                del self._batches[batch_id]
                return True
            return False

    def list_batches(
        self,
        tenant_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[BatchProvenance]:
        """
        List batch provenance records with optional filtering.

        Args:
            tenant_id: Optional tenant ID to filter by.
            limit: Maximum number of batches to return.
            offset: Number of batches to skip.

        Returns:
            List of BatchProvenance objects.
        """
        with self._lock:
            all_batches = list(self._batches.values())
            if tenant_id is not None:
                all_batches = [b for b in all_batches if b.tenant_id == tenant_id]
            all_batches.sort(key=lambda b: b.batch_started_at)
            return all_batches[offset: offset + limit]

    def reset(self) -> None:
        """
        Clear all batches (for testing).
        """
        with self._lock:
            self._batches.clear()


# ============================================================================
# STANDALONE HASH FUNCTIONS (20+)
# ============================================================================


def hash_calculation_input(input_data: Dict[str, Any]) -> str:
    """
    Hash complete calculation request.

    Args:
        input_data: Calculation request dictionary containing employee
                    commute data, method selection, and configuration.

    Returns:
        SHA-256 hash (64-character hex string).
    """
    return _compute_hash(input_data)


def hash_commute_input(commute_input: Dict[str, Any]) -> str:
    """
    Hash commute pattern input data.

    Covers the overall commute input record for an employee including
    mode, distance, frequency, and working pattern information.

    Args:
        commute_input: Commute data dictionary with keys such as
            mode, vehicle_type, distance_km, frequency,
            working_days_per_week, one_way_or_round_trip, etc.

    Returns:
        SHA-256 hash (64-character hex string).

    Example:
        >>> h = hash_commute_input({
        ...     "mode": "car",
        ...     "distance_km": 25.0,
        ...     "working_days": 230
        ... })
    """
    return _compute_hash(commute_input)


def hash_vehicle_input(vehicle_input: Dict[str, Any]) -> str:
    """
    Hash vehicle-specific commute input data.

    For employees commuting by personal vehicle (car, motorcycle, etc.).
    Captures vehicle characteristics that determine the emission factor.

    Args:
        vehicle_input: Vehicle data dictionary with keys such as
            vehicle_type (small_car, medium_car, large_car, suv,
            motorcycle, electric_vehicle, hybrid, plug_in_hybrid),
            fuel_type (petrol, diesel, lpg, cng, electric, hybrid),
            engine_size_cc, vehicle_year, annual_mileage_km,
            occupancy, etc.

    Returns:
        SHA-256 hash (64-character hex string).

    Example:
        >>> h = hash_vehicle_input({
        ...     "vehicle_type": "medium_car",
        ...     "fuel_type": "petrol",
        ...     "engine_size_cc": 1600
        ... })
    """
    return _compute_hash(vehicle_input)


def hash_transit_input(transit_input: Dict[str, Any]) -> str:
    """
    Hash public transit commute input data.

    For employees commuting by bus, rail, subway, tram, ferry, or
    other public transit modes.

    Args:
        transit_input: Transit data dictionary with keys such as
            transit_mode (bus, rail, subway, tram, light_rail, ferry,
            commuter_rail), route_distance_km, fare_zone, frequency,
            pass_type, transit_agency, etc.

    Returns:
        SHA-256 hash (64-character hex string).

    Example:
        >>> h = hash_transit_input({
        ...     "transit_mode": "commuter_rail",
        ...     "route_distance_km": 45.0,
        ...     "frequency": "daily"
        ... })
    """
    return _compute_hash(transit_input)


def hash_telework_input(telework_input: Dict[str, Any]) -> str:
    """
    Hash telework/remote work input data.

    For employees working from home. Captures the home office energy
    parameters needed to compute telework emissions.

    Args:
        telework_input: Telework data dictionary with keys such as
            days_per_week, country, region, home_energy_type
            (grid_electricity, natural_gas, heat_pump),
            grid_emission_factor, daily_kwh_estimate,
            home_office_sqft, heating_type, cooling_type, etc.

    Returns:
        SHA-256 hash (64-character hex string).

    Example:
        >>> h = hash_telework_input({
        ...     "days_per_week": 2,
        ...     "country": "US",
        ...     "daily_kwh": 3.5
        ... })
    """
    return _compute_hash(telework_input)


def hash_carpool_input(carpool_input: Dict[str, Any]) -> str:
    """
    Hash carpool/vanpool commute input data.

    For employees sharing vehicles. The allocation factor divides
    vehicle emissions among occupants.

    Args:
        carpool_input: Carpool data dictionary with keys such as
            vehicle_type, fuel_type, total_distance_km, occupants,
            driver_employee_id, allocation_method (equal, distance_weighted),
            carpool_days_per_week, etc.

    Returns:
        SHA-256 hash (64-character hex string).

    Example:
        >>> h = hash_carpool_input({
        ...     "vehicle_type": "medium_car",
        ...     "occupants": 4,
        ...     "total_distance_km": 30.0
        ... })
    """
    return _compute_hash(carpool_input)


def hash_survey_input(survey_input: Dict[str, Any]) -> str:
    """
    Hash employee commute survey input data.

    For survey-based calculation methods where a sample of employees
    provides commute data that is extrapolated to the full population.

    Args:
        survey_input: Survey data dictionary with keys such as
            survey_id, survey_year, total_respondents, response_rate,
            total_population, stratification_method, weighting_scheme,
            confidence_level, margin_of_error, etc.

    Returns:
        SHA-256 hash (64-character hex string).

    Example:
        >>> h = hash_survey_input({
        ...     "survey_year": 2025,
        ...     "total_respondents": 4250,
        ...     "total_population": 10000,
        ...     "response_rate": 0.85
        ... })
    """
    return _compute_hash(survey_input)


def hash_average_data_input(avg_input: Dict[str, Any]) -> str:
    """
    Hash average-data method input.

    For the average-data calculation method where national or regional
    average commuting statistics are used instead of employee-specific data.

    Args:
        avg_input: Average data dictionary with keys such as
            country, region, employee_count, average_commute_distance_km,
            average_working_days, modal_split (dict of mode percentages),
            data_source, data_year, etc.

    Returns:
        SHA-256 hash (64-character hex string).

    Example:
        >>> h = hash_average_data_input({
        ...     "country": "US",
        ...     "employee_count": 10000,
        ...     "average_commute_distance_km": 16.0,
        ...     "data_source": "ACS"
        ... })
    """
    return _compute_hash(avg_input)


def hash_spend_input(spend_input: Dict[str, Any]) -> str:
    """
    Hash spend-based calculation input.

    For the spend-based calculation method where transit/fuel expenditure
    data is used with EEIO emission factors.

    Args:
        spend_input: Spend data dictionary with keys such as
            spend_amount, currency, spend_category (fuel, transit_pass,
            parking), reporting_period, eeio_sector, deflation_factor,
            margin_removal_factor, etc.

    Returns:
        SHA-256 hash (64-character hex string).

    Example:
        >>> h = hash_spend_input({
        ...     "spend_amount": 2500.00,
        ...     "currency": "USD",
        ...     "spend_category": "fuel"
        ... })
    """
    return _compute_hash(spend_input)


def hash_emission_factor(ef_data: Dict[str, Any]) -> str:
    """
    Hash emission factor data.

    Hashes the complete emission factor record including type, value,
    source, year, region, and any other qualifying attributes.

    Args:
        ef_data: Emission factor dictionary with keys such as
            ef_type (e.g., "car_medium_petrol", "bus_local_diesel"),
            ef_value (kgCO2e per unit), source (e.g., "DEFRA", "EPA"),
            year, region, unit, gas_breakdown (co2, ch4, n2o), etc.

    Returns:
        SHA-256 hash (64-character hex string).

    Example:
        >>> h = hash_emission_factor({
        ...     "ef_type": "car_medium_petrol",
        ...     "ef_value": "0.17048",
        ...     "source": "DEFRA",
        ...     "year": 2025
        ... })
    """
    return _compute_hash(ef_data)


def hash_calculation_result(result: Dict[str, Any]) -> str:
    """
    Hash complete calculation result.

    Args:
        result: Calculation result dictionary with all emissions
                components, totals, and metadata.

    Returns:
        SHA-256 hash (64-character hex string).
    """
    return _compute_hash(result)


def hash_commute_result(result: Dict[str, Any]) -> str:
    """
    Hash commute transport calculation result.

    Hashes the output of the commute emissions calculation engine
    for a single employee or commute pattern.

    Args:
        result: Commute result dictionary with keys such as
            co2e_kg, co2_kg, ch4_kg, n2o_kg, mode, method,
            distance_km, working_days, emission_factor_used, etc.

    Returns:
        SHA-256 hash (64-character hex string).

    Example:
        >>> h = hash_commute_result({
        ...     "co2e_kg": 1825.0,
        ...     "mode": "car",
        ...     "method": "distance_based"
        ... })
    """
    return _compute_hash(result)


def hash_vehicle_result(result: Dict[str, Any]) -> str:
    """
    Hash vehicle-specific emissions calculation result.

    Args:
        result: Vehicle result dictionary with keys such as
            co2e_kg, vehicle_type, fuel_type, distance_km,
            ef_used, occupancy_factor, annual_working_days, etc.

    Returns:
        SHA-256 hash (64-character hex string).

    Example:
        >>> h = hash_vehicle_result({
        ...     "co2e_kg": 1460.0,
        ...     "vehicle_type": "medium_car",
        ...     "fuel_type": "petrol"
        ... })
    """
    return _compute_hash(result)


def hash_transit_result(result: Dict[str, Any]) -> str:
    """
    Hash public transit emissions calculation result.

    Args:
        result: Transit result dictionary with keys such as
            co2e_kg, transit_mode, route_distance_km,
            ef_used, annual_trips, etc.

    Returns:
        SHA-256 hash (64-character hex string).

    Example:
        >>> h = hash_transit_result({
        ...     "co2e_kg": 365.0,
        ...     "transit_mode": "commuter_rail",
        ...     "route_distance_km": 45.0
        ... })
    """
    return _compute_hash(result)


def hash_telework_result(result: Dict[str, Any]) -> str:
    """
    Hash telework emissions calculation result.

    Args:
        result: Telework result dictionary with keys such as
            co2e_kg, country, annual_telework_days, daily_kwh,
            grid_ef_kgco2e_per_kwh, heating_co2e_kg, cooling_co2e_kg,
            equipment_co2e_kg, etc.

    Returns:
        SHA-256 hash (64-character hex string).

    Example:
        >>> h = hash_telework_result({
        ...     "co2e_kg": 156.0,
        ...     "country": "US",
        ...     "annual_telework_days": 104
        ... })
    """
    return _compute_hash(result)


def hash_compliance_result(result: Dict[str, Any]) -> str:
    """
    Hash compliance check result.

    Args:
        result: Compliance result dictionary with keys such as
            framework (GHG_PROTOCOL, CSRD, CDP, SBTi, ISO_14064),
            status (COMPLIANT, NON_COMPLIANT, PARTIAL),
            score (0.0-1.0), findings, recommendations, etc.

    Returns:
        SHA-256 hash (64-character hex string).

    Example:
        >>> h = hash_compliance_result({
        ...     "framework": "GHG_PROTOCOL",
        ...     "status": "COMPLIANT",
        ...     "score": 0.95
        ... })
    """
    return _compute_hash(result)


def hash_aggregation_result(result: Dict[str, Any]) -> str:
    """
    Hash aggregation and summarization result.

    Args:
        result: Aggregation result dictionary with keys such as
            total_co2e_kg, by_mode, by_department, by_location,
            by_working_pattern, employee_count, average_per_employee,
            modal_split, etc.

    Returns:
        SHA-256 hash (64-character hex string).

    Example:
        >>> h = hash_aggregation_result({
        ...     "total_co2e_kg": 1250000.0,
        ...     "employee_count": 10000,
        ...     "by_mode": {"car": 750000.0, "transit": 350000.0}
        ... })
    """
    return _compute_hash(result)


def hash_multi_modal_input(input_data: Dict[str, Any]) -> str:
    """
    Hash multi-modal commute input data.

    For employees who use multiple transport modes in a single commute
    (e.g., drive to station, then take train). Each leg of the commute
    is captured separately.

    Args:
        input_data: Multi-modal data dictionary with keys such as
            legs (list of leg dictionaries, each with mode, distance_km,
            vehicle_type, transit_mode), total_distance_km,
            transfer_points, etc.

    Returns:
        SHA-256 hash (64-character hex string).

    Example:
        >>> h = hash_multi_modal_input({
        ...     "legs": [
        ...         {"mode": "car", "distance_km": 5.0},
        ...         {"mode": "rail", "distance_km": 40.0}
        ...     ],
        ...     "total_distance_km": 45.0
        ... })
    """
    return _compute_hash(input_data)


def hash_batch_input(batch_items: List[Dict[str, Any]]) -> str:
    """
    Hash a batch of input items.

    Computes a single hash over an entire batch of employee commute
    input records. The list is serialized deterministically.

    Args:
        batch_items: List of input item dictionaries.

    Returns:
        SHA-256 hash (64-character hex string).

    Example:
        >>> h = hash_batch_input([
        ...     {"employee_id": "E001", "mode": "car"},
        ...     {"employee_id": "E002", "mode": "bus"}
        ... ])
    """
    return _compute_hash(batch_items)


# ============================================================================
# ADDITIONAL DOMAIN-SPECIFIC HASH FUNCTIONS
# ============================================================================


def hash_distance_result(
    distance_km: Union[Decimal, float],
    mode: str,
    ef: Union[Decimal, float],
    working_days: int,
    co2e_kg: Union[Decimal, float],
) -> str:
    """
    Hash distance-based calculation result.

    Args:
        distance_km: One-way commute distance in kilometres.
        mode: Commute mode.
        ef: Emission factor (kgCO2e/passenger-km).
        working_days: Annual working days (commuting days).
        co2e_kg: Total annual emissions (kgCO2e).

    Returns:
        SHA-256 hash (64-character hex string).
    """
    data = {
        "distance_km": str(distance_km),
        "mode": mode,
        "ef": str(ef),
        "working_days": working_days,
        "co2e_kg": str(co2e_kg),
        "formula": "distance_km * 2 * working_days * ef",
    }
    return _compute_hash(data)


def hash_spend_result(
    spend: Union[Decimal, float],
    currency: str,
    mode: str,
    ef: Union[Decimal, float],
    co2e_kg: Union[Decimal, float],
) -> str:
    """
    Hash spend-based calculation result.

    Args:
        spend: Annual transit spend in original currency.
        currency: ISO 4217 currency code.
        mode: Commute mode.
        ef: Emission factor (kgCO2e per currency unit).
        co2e_kg: Total emissions (kgCO2e).

    Returns:
        SHA-256 hash (64-character hex string).
    """
    data = {
        "spend": str(spend),
        "currency": currency,
        "mode": mode,
        "ef": str(ef),
        "co2e_kg": str(co2e_kg),
        "formula": "spend * ef",
    }
    return _compute_hash(data)


def hash_telework_calc_result(
    country: str,
    days: int,
    ef_kwh: Union[Decimal, float],
    daily_kwh: Union[Decimal, float],
    co2e_kg: Union[Decimal, float],
) -> str:
    """
    Hash telework emissions calculation result (parametric form).

    Args:
        country: ISO 3166-1 alpha-2 country code.
        days: Annual telework days.
        ef_kwh: Grid emission factor (kgCO2e/kWh).
        daily_kwh: Daily home office energy consumption (kWh).
        co2e_kg: Total telework emissions (kgCO2e).

    Returns:
        SHA-256 hash (64-character hex string).
    """
    data = {
        "country": country,
        "days": days,
        "ef_kwh": str(ef_kwh),
        "daily_kwh": str(daily_kwh),
        "co2e_kg": str(co2e_kg),
        "formula": "days * daily_kwh * ef_kwh",
    }
    return _compute_hash(data)


def hash_extrapolation_result(
    sample_size: int,
    total_population: int,
    response_rate: Union[Decimal, float],
    sample_co2e_kg: Union[Decimal, float],
    extrapolated_co2e_kg: Union[Decimal, float],
) -> str:
    """
    Hash population extrapolation result.

    Args:
        sample_size: Number of surveyed employees.
        total_population: Total employee population.
        response_rate: Survey response rate (0.0 to 1.0).
        sample_co2e_kg: Total emissions from sample (kgCO2e).
        extrapolated_co2e_kg: Extrapolated total emissions (kgCO2e).

    Returns:
        SHA-256 hash (64-character hex string).
    """
    data = {
        "sample_size": sample_size,
        "total_population": total_population,
        "response_rate": str(response_rate),
        "sample_co2e_kg": str(sample_co2e_kg),
        "extrapolated_co2e_kg": str(extrapolated_co2e_kg),
        "formula": "(sample_co2e / sample_size) * total_population",
    }
    return _compute_hash(data)


def hash_classification_result(
    mode: str, vehicle_type: str, working_pattern: str
) -> str:
    """
    Hash commute classification result.

    Args:
        mode: Commute mode (e.g., "car", "bus", "rail", "bicycle").
        vehicle_type: Vehicle type (e.g., "medium_car", "bus_local").
        working_pattern: Working pattern (e.g., "full_time_office",
                         "hybrid_3_2", "full_remote").

    Returns:
        SHA-256 hash (64-character hex string).
    """
    data = {
        "mode": mode,
        "vehicle_type": vehicle_type,
        "working_pattern": working_pattern,
    }
    return _compute_hash(data)


def hash_allocation_result(
    method: str,
    share: Union[Decimal, float],
    total: Union[Decimal, float],
    allocated: Union[Decimal, float],
) -> str:
    """
    Hash allocation calculation result.

    Args:
        method: Allocation method (e.g., "headcount", "fte",
                "office_location", "department").
        share: This unit's share (e.g., 0.25 for 25%).
        total: Total emissions before allocation.
        allocated: Allocated emissions.

    Returns:
        SHA-256 hash (64-character hex string).
    """
    data = {
        "method": method,
        "share": str(share),
        "total": str(total),
        "allocated": str(allocated),
        "formula": "total * share",
    }
    return _compute_hash(data)


def hash_aggregation(
    by_mode: Dict[str, Any],
    by_department: Dict[str, Any],
    by_location: Dict[str, Any],
    total: Union[Decimal, float],
) -> str:
    """
    Hash aggregation result (parametric form).

    Args:
        by_mode: Emissions by commute mode.
        by_department: Emissions by department.
        by_location: Emissions by office location.
        total: Total emissions (kgCO2e).

    Returns:
        SHA-256 hash (64-character hex string).
    """
    data = {
        "by_mode": by_mode,
        "by_department": by_department,
        "by_location": by_location,
        "total": str(total),
    }
    return _compute_hash(data)


def hash_uncertainty_result(
    uncertainty_type: str,
    confidence_level: Union[Decimal, float],
    lower_bound: Union[Decimal, float],
    upper_bound: Union[Decimal, float],
) -> str:
    """
    Hash uncertainty quantification result.

    Args:
        uncertainty_type: Type of uncertainty (e.g., "parametric", "model",
                          "survey_sampling").
        confidence_level: Confidence level (e.g., 0.95 for 95%).
        lower_bound: Lower confidence bound.
        upper_bound: Upper confidence bound.

    Returns:
        SHA-256 hash (64-character hex string).
    """
    data = {
        "uncertainty_type": uncertainty_type,
        "confidence_level": str(confidence_level),
        "lower_bound": str(lower_bound),
        "upper_bound": str(upper_bound),
    }
    return _compute_hash(data)


def hash_dqi_result(
    dqi_scores: Dict[str, Union[Decimal, float]],
    composite: Union[Decimal, float],
) -> str:
    """
    Hash data quality indicator assessment.

    Args:
        dqi_scores: DQI scores by dimension (e.g., reliability, completeness,
                    temporal_correlation, geographical_correlation,
                    technological_correlation).
        composite: Composite DQI score.

    Returns:
        SHA-256 hash (64-character hex string).
    """
    data = {
        "dqi_scores": {k: str(v) for k, v in dqi_scores.items()},
        "composite": str(composite),
    }
    return _compute_hash(data)


def hash_modal_split(
    modal_shares: Dict[str, Union[Decimal, float]],
    total_employees: int,
    total_co2e_kg: Union[Decimal, float],
) -> str:
    """
    Hash modal split analysis result.

    Args:
        modal_shares: Percentage share by commute mode.
        total_employees: Total employee count.
        total_co2e_kg: Total emissions (kgCO2e).

    Returns:
        SHA-256 hash (64-character hex string).
    """
    data = {
        "modal_shares": {k: str(v) for k, v in modal_shares.items()},
        "total_employees": total_employees,
        "total_co2e_kg": str(total_co2e_kg),
    }
    return _compute_hash(data)


def hash_batch_result(batch_id: str, results: List[Dict[str, Any]]) -> str:
    """
    Hash batch calculation result.

    Args:
        batch_id: Batch identifier.
        results: List of individual employee results.

    Returns:
        SHA-256 hash (64-character hex string).
    """
    data = {"batch_id": batch_id, "results": results}
    return _compute_hash(data)


def hash_employee_input(employee: Dict[str, Any]) -> str:
    """
    Hash employee-level input data.

    Args:
        employee: Employee data (commute mode, distance, working days,
                  telework frequency, location, department, etc.)

    Returns:
        SHA-256 hash (64-character hex string).
    """
    return _compute_hash(employee)


def hash_config(config: Dict[str, Any]) -> str:
    """
    Hash configuration data.

    Args:
        config: Configuration dictionary.

    Returns:
        SHA-256 hash (64-character hex string).
    """
    return _compute_hash(config)


def hash_metadata(metadata: Dict[str, Any]) -> str:
    """
    Hash metadata.

    Args:
        metadata: Metadata dictionary.

    Returns:
        SHA-256 hash (64-character hex string).
    """
    return _compute_hash(metadata)


def hash_arbitrary(data: Any) -> str:
    """
    Hash arbitrary data.

    Args:
        data: Any data to hash.

    Returns:
        SHA-256 hash (64-character hex string).
    """
    return _compute_hash(data)


# ============================================================================
# SINGLETON ACCESS
# ============================================================================

_tracker_instance: Optional[ProvenanceTracker] = None
_tracker_lock = threading.RLock()

_batch_tracker_instance: Optional[BatchProvenanceTracker] = None
_batch_tracker_lock = threading.RLock()


def get_provenance_tracker() -> ProvenanceTracker:
    """
    Get singleton ProvenanceTracker instance.

    Thread-safe singleton pattern. Returns the same tracker instance
    across all callers in the process.

    Returns:
        ProvenanceTracker instance.

    Example:
        >>> tracker = get_provenance_tracker()
        >>> chain = tracker.start_chain("tenant-001")
    """
    global _tracker_instance
    with _tracker_lock:
        if _tracker_instance is None:
            _tracker_instance = ProvenanceTracker()
        return _tracker_instance


def reset_provenance_tracker() -> None:
    """
    Reset singleton ProvenanceTracker instance.

    Useful for testing. Creates a fresh tracker on next access.
    Also resets the batch tracker since it depends on the main tracker.

    Example:
        >>> reset_provenance_tracker()
        >>> tracker = get_provenance_tracker()  # Fresh instance
    """
    global _tracker_instance, _batch_tracker_instance
    with _tracker_lock:
        _tracker_instance = None
    with _batch_tracker_lock:
        _batch_tracker_instance = None


def get_batch_provenance_tracker() -> BatchProvenanceTracker:
    """
    Get singleton BatchProvenanceTracker instance.

    Thread-safe singleton pattern. The batch tracker is initialized with
    the singleton ProvenanceTracker instance.

    Returns:
        BatchProvenanceTracker instance.

    Example:
        >>> batch_tracker = get_batch_provenance_tracker()
        >>> batch_id = batch_tracker.start_batch("tenant-001", 5000)
    """
    global _batch_tracker_instance
    with _batch_tracker_lock:
        if _batch_tracker_instance is None:
            tracker = get_provenance_tracker()
            _batch_tracker_instance = BatchProvenanceTracker(tracker)
        return _batch_tracker_instance


def reset_batch_provenance_tracker() -> None:
    """
    Reset singleton BatchProvenanceTracker instance.

    Useful for testing. Creates a fresh batch tracker on next access.

    Example:
        >>> reset_batch_provenance_tracker()
        >>> batch_tracker = get_batch_provenance_tracker()
    """
    global _batch_tracker_instance
    with _batch_tracker_lock:
        _batch_tracker_instance = None


# ============================================================================
# CONVENIENCE FUNCTIONS (10 stage-specific recorders)
# ============================================================================


def create_chain(
    tenant_id: str = "default",
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[ProvenanceTracker, str]:
    """
    Create a new provenance chain using singleton tracker.

    Args:
        tenant_id: Tenant identifier (default: "default").
        metadata: Optional creation metadata.

    Returns:
        Tuple of (tracker, chain_id).

    Example:
        >>> tracker, chain_id = create_chain("tenant-001")
        >>> tracker.record_stage(chain_id, "VALIDATE", inp, out)
    """
    tracker = get_provenance_tracker()
    chain = tracker.start_chain(tenant_id, metadata)
    return tracker, chain.chain_id


def record_validate_stage(
    chain_id: str,
    input_data: Any,
    validated_data: Any,
    engine_id: str = "validation-engine",
    engine_version: str = AGENT_VERSION,
    metadata: Optional[Dict[str, Any]] = None,
    duration_ms: float = 0.0,
) -> ProvenanceEntry:
    """
    Record VALIDATE stage.

    Records input validation and data quality checks for employee
    commuting data.

    Args:
        chain_id: Chain identifier.
        input_data: Raw input data (employee commute records).
        validated_data: Validated data (cleaned, type-checked records).
        engine_id: Engine identifier.
        engine_version: Engine version.
        metadata: Optional metadata (e.g., rule_count, pass_rate, errors).
        duration_ms: Processing time in milliseconds.

    Returns:
        ProvenanceEntry.
    """
    tracker = get_provenance_tracker()
    return tracker.record_stage(
        chain_id, ProvenanceStage.VALIDATE, input_data, validated_data,
        engine_id=engine_id, engine_version=engine_version,
        metadata=metadata, duration_ms=duration_ms,
    )


def record_classify_stage(
    chain_id: str,
    input_data: Any,
    classified_data: Any,
    engine_id: str = "classification-engine",
    engine_version: str = AGENT_VERSION,
    metadata: Optional[Dict[str, Any]] = None,
    duration_ms: float = 0.0,
) -> ProvenanceEntry:
    """
    Record CLASSIFY stage.

    Records commute classification results including transport mode,
    vehicle type, and working pattern.

    Args:
        chain_id: Chain identifier.
        input_data: Input data (validated employee records).
        classified_data: Classified data (mode, vehicle_type, working_pattern).
        engine_id: Engine identifier.
        engine_version: Engine version.
        metadata: Optional metadata.
        duration_ms: Processing time in milliseconds.

    Returns:
        ProvenanceEntry.
    """
    tracker = get_provenance_tracker()
    return tracker.record_stage(
        chain_id, ProvenanceStage.CLASSIFY, input_data, classified_data,
        engine_id=engine_id, engine_version=engine_version,
        metadata=metadata, duration_ms=duration_ms,
    )


def record_normalize_stage(
    chain_id: str,
    input_data: Any,
    normalized_data: Any,
    engine_id: str = "normalization-engine",
    engine_version: str = AGENT_VERSION,
    metadata: Optional[Dict[str, Any]] = None,
    duration_ms: float = 0.0,
) -> ProvenanceEntry:
    """
    Record NORMALIZE stage.

    Records unit normalization results: distances to km, currencies to
    reporting currency, working days to annual equivalents.

    Args:
        chain_id: Chain identifier.
        input_data: Input data (classified records).
        normalized_data: Normalized data (standardized units).
        engine_id: Engine identifier.
        engine_version: Engine version.
        metadata: Optional metadata.
        duration_ms: Processing time in milliseconds.

    Returns:
        ProvenanceEntry.
    """
    tracker = get_provenance_tracker()
    return tracker.record_stage(
        chain_id, ProvenanceStage.NORMALIZE, input_data, normalized_data,
        engine_id=engine_id, engine_version=engine_version,
        metadata=metadata, duration_ms=duration_ms,
    )


def record_resolve_efs_stage(
    chain_id: str,
    input_data: Any,
    resolved_efs: Any,
    engine_id: str = "ef-resolution-engine",
    engine_version: str = AGENT_VERSION,
    metadata: Optional[Dict[str, Any]] = None,
    duration_ms: float = 0.0,
) -> ProvenanceEntry:
    """
    Record RESOLVE_EFS stage.

    Records emission factor resolution from DEFRA, EPA, IEA, or other
    sources based on commute mode, vehicle type, fuel type, and region.

    Args:
        chain_id: Chain identifier.
        input_data: Input data (mode, vehicle type, region, etc.).
        resolved_efs: Resolved emission factors with sources.
        engine_id: Engine identifier.
        engine_version: Engine version.
        metadata: Optional metadata (e.g., ef_source, ef_year).
        duration_ms: Processing time in milliseconds.

    Returns:
        ProvenanceEntry.
    """
    tracker = get_provenance_tracker()
    return tracker.record_stage(
        chain_id, ProvenanceStage.RESOLVE_EFS, input_data, resolved_efs,
        engine_id=engine_id, engine_version=engine_version,
        metadata=metadata, duration_ms=duration_ms,
    )


def record_calculate_commute_stage(
    chain_id: str,
    input_data: Any,
    commute_results: Any,
    engine_id: str = "commute-calculation-engine",
    engine_version: str = AGENT_VERSION,
    metadata: Optional[Dict[str, Any]] = None,
    duration_ms: float = 0.0,
) -> ProvenanceEntry:
    """
    Record CALCULATE_COMMUTE stage.

    Records commute transport emissions calculation applying distance-based,
    spend-based, or average-data methods.

    Args:
        chain_id: Chain identifier.
        input_data: Commute calculation input (distances, EFs, working days).
        commute_results: Transport calculation results (CO2e per mode).
        engine_id: Engine identifier.
        engine_version: Engine version.
        metadata: Optional metadata (e.g., method, formula).
        duration_ms: Processing time in milliseconds.

    Returns:
        ProvenanceEntry.
    """
    tracker = get_provenance_tracker()
    return tracker.record_stage(
        chain_id, ProvenanceStage.CALCULATE_COMMUTE, input_data, commute_results,
        engine_id=engine_id, engine_version=engine_version,
        metadata=metadata, duration_ms=duration_ms,
    )


def record_calculate_telework_stage(
    chain_id: str,
    input_data: Any,
    telework_results: Any,
    engine_id: str = "telework-calculation-engine",
    engine_version: str = AGENT_VERSION,
    metadata: Optional[Dict[str, Any]] = None,
    duration_ms: float = 0.0,
) -> ProvenanceEntry:
    """
    Record CALCULATE_TELEWORK stage.

    Records telework/remote work emissions calculation including
    home office energy emissions.

    Args:
        chain_id: Chain identifier.
        input_data: Telework calculation input (days, country, grid EF).
        telework_results: Telework calculation results (CO2e).
        engine_id: Engine identifier.
        engine_version: Engine version.
        metadata: Optional metadata.
        duration_ms: Processing time in milliseconds.

    Returns:
        ProvenanceEntry.
    """
    tracker = get_provenance_tracker()
    return tracker.record_stage(
        chain_id, ProvenanceStage.CALCULATE_TELEWORK, input_data, telework_results,
        engine_id=engine_id, engine_version=engine_version,
        metadata=metadata, duration_ms=duration_ms,
    )


def record_apply_allocation_stage(
    chain_id: str,
    input_data: Any,
    allocation_results: Any,
    engine_id: str = "allocation-engine",
    engine_version: str = AGENT_VERSION,
    metadata: Optional[Dict[str, Any]] = None,
    duration_ms: float = 0.0,
) -> ProvenanceEntry:
    """
    Record APPLY_ALLOCATION stage.

    Records allocation of emissions to reporting entities using
    headcount, FTE, office location, or department-based methods.

    Args:
        chain_id: Chain identifier.
        input_data: Pre-allocation data and allocation parameters.
        allocation_results: Allocated emissions by entity.
        engine_id: Engine identifier.
        engine_version: Engine version.
        metadata: Optional metadata (e.g., allocation_method, shares).
        duration_ms: Processing time in milliseconds.

    Returns:
        ProvenanceEntry.
    """
    tracker = get_provenance_tracker()
    return tracker.record_stage(
        chain_id, ProvenanceStage.APPLY_ALLOCATION, input_data, allocation_results,
        engine_id=engine_id, engine_version=engine_version,
        metadata=metadata, duration_ms=duration_ms,
    )


def record_compliance_stage(
    chain_id: str,
    input_data: Any,
    compliance_results: Any,
    engine_id: str = "compliance-engine",
    engine_version: str = AGENT_VERSION,
    metadata: Optional[Dict[str, Any]] = None,
    duration_ms: float = 0.0,
) -> ProvenanceEntry:
    """
    Record COMPLIANCE stage.

    Records regulatory compliance check results against GHG Protocol,
    CSRD, CDP, SBTi, and other applicable frameworks.

    Args:
        chain_id: Chain identifier.
        input_data: Data to check against compliance rules.
        compliance_results: Compliance check results per framework.
        engine_id: Engine identifier.
        engine_version: Engine version.
        metadata: Optional metadata (e.g., frameworks_checked).
        duration_ms: Processing time in milliseconds.

    Returns:
        ProvenanceEntry.
    """
    tracker = get_provenance_tracker()
    return tracker.record_stage(
        chain_id, ProvenanceStage.COMPLIANCE, input_data, compliance_results,
        engine_id=engine_id, engine_version=engine_version,
        metadata=metadata, duration_ms=duration_ms,
    )


def record_aggregate_stage(
    chain_id: str,
    input_data: Any,
    aggregated_results: Any,
    engine_id: str = "aggregation-engine",
    engine_version: str = AGENT_VERSION,
    metadata: Optional[Dict[str, Any]] = None,
    duration_ms: float = 0.0,
) -> ProvenanceEntry:
    """
    Record AGGREGATE stage.

    Records aggregation and summarization by commute mode, department,
    office location, working pattern, and other dimensions.

    Args:
        chain_id: Chain identifier.
        input_data: Pre-aggregation data (per-employee results).
        aggregated_results: Aggregated results (totals by dimension).
        engine_id: Engine identifier.
        engine_version: Engine version.
        metadata: Optional metadata (e.g., dimensions).
        duration_ms: Processing time in milliseconds.

    Returns:
        ProvenanceEntry.
    """
    tracker = get_provenance_tracker()
    return tracker.record_stage(
        chain_id, ProvenanceStage.AGGREGATE, input_data, aggregated_results,
        engine_id=engine_id, engine_version=engine_version,
        metadata=metadata, duration_ms=duration_ms,
    )


def record_seal_stage(
    chain_id: str,
    input_data: Any,
    seal_data: Any,
    engine_id: str = "seal-engine",
    engine_version: str = AGENT_VERSION,
    metadata: Optional[Dict[str, Any]] = None,
    duration_ms: float = 0.0,
) -> ProvenanceEntry:
    """
    Record SEAL stage.

    Records the final sealing step before the chain is formally sealed.
    This stage captures the final state summary and any seal-specific
    metadata before the chain becomes immutable.

    Args:
        chain_id: Chain identifier.
        input_data: Pre-seal chain summary.
        seal_data: Seal verification data.
        engine_id: Engine identifier.
        engine_version: Engine version.
        metadata: Optional metadata (e.g., seal_reason, auditor_id).
        duration_ms: Processing time in milliseconds.

    Returns:
        ProvenanceEntry.
    """
    tracker = get_provenance_tracker()
    return tracker.record_stage(
        chain_id, ProvenanceStage.SEAL, input_data, seal_data,
        engine_id=engine_id, engine_version=engine_version,
        metadata=metadata, duration_ms=duration_ms,
    )


def seal_and_verify(chain_id: str) -> Tuple[str, bool]:
    """
    Seal chain and verify integrity.

    Convenience function that seals the chain and immediately validates
    it, returning both the final hash and validation status.

    Args:
        chain_id: Chain identifier.

    Returns:
        Tuple of (final_hash, is_valid).

    Example:
        >>> final_hash, is_valid = seal_and_verify(chain_id)
        >>> assert is_valid, "Chain integrity check failed"
    """
    tracker = get_provenance_tracker()
    final_hash = tracker.seal_chain(chain_id)
    is_valid, _errors = tracker.validate_chain(chain_id)
    return final_hash, is_valid


def export_chain_json(chain_id: str) -> str:
    """
    Export chain as JSON.

    Args:
        chain_id: Chain identifier.

    Returns:
        JSON string representation of the complete chain.

    Example:
        >>> json_str = export_chain_json(chain_id)
        >>> chain_dict = json.loads(json_str)
    """
    tracker = get_provenance_tracker()
    chain_dict = tracker.export_chain(chain_id)
    return json.dumps(chain_dict, indent=2, sort_keys=True, default=str)


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "ProvenanceStage",
    # Models
    "ProvenanceEntry",
    "ProvenanceChain",
    "BatchProvenance",
    # Trackers
    "ProvenanceTracker",
    "BatchProvenanceTracker",
    # Singleton accessors
    "get_provenance_tracker",
    "reset_provenance_tracker",
    "get_batch_provenance_tracker",
    "reset_batch_provenance_tracker",
    # Hash utilities (internal but exported for testing)
    "_canonical_json",
    "_sha256",
    "_serialize",
    "_compute_hash",
    "_compute_chain_hash",
    "_merkle_hash",
    "_merkle_proof",
    "_verify_merkle_proof",
    # Standalone hash functions (20+)
    "hash_calculation_input",
    "hash_commute_input",
    "hash_vehicle_input",
    "hash_transit_input",
    "hash_telework_input",
    "hash_carpool_input",
    "hash_survey_input",
    "hash_average_data_input",
    "hash_spend_input",
    "hash_emission_factor",
    "hash_calculation_result",
    "hash_commute_result",
    "hash_vehicle_result",
    "hash_transit_result",
    "hash_telework_result",
    "hash_compliance_result",
    "hash_aggregation_result",
    "hash_multi_modal_input",
    "hash_batch_input",
    # Domain-specific parametric hash functions
    "hash_distance_result",
    "hash_spend_result",
    "hash_telework_calc_result",
    "hash_extrapolation_result",
    "hash_classification_result",
    "hash_allocation_result",
    "hash_aggregation",
    "hash_uncertainty_result",
    "hash_dqi_result",
    "hash_modal_split",
    "hash_batch_result",
    "hash_employee_input",
    "hash_config",
    "hash_metadata",
    "hash_arbitrary",
    # Convenience functions (10 stage recorders + helpers)
    "create_chain",
    "record_validate_stage",
    "record_classify_stage",
    "record_normalize_stage",
    "record_resolve_efs_stage",
    "record_calculate_commute_stage",
    "record_calculate_telework_stage",
    "record_apply_allocation_stage",
    "record_compliance_stage",
    "record_aggregate_stage",
    "record_seal_stage",
    "seal_and_verify",
    "export_chain_json",
]
