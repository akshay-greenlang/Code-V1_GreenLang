# -*- coding: utf-8 -*-
"""
End-of-Life Treatment of Sold Products Provenance Tracking - AGENT-MRV-025

SHA-256 chain-hashed provenance for zero-hallucination audit trails.
10 pipeline stages tracked with deterministic hashing.
Agent: GL-MRV-S3-012

This module implements a complete provenance tracking system for end-of-life
treatment of sold products emissions calculations. Every step in the
calculation pipeline is recorded with SHA-256 hashes, creating an immutable
audit trail that proves no data was hallucinated or modified.

Pipeline Stages:
    1. VALIDATE - Input validation and data quality checks
    2. CLASSIFY - Product material composition and treatment classification
    3. NORMALIZE - Unit normalization (mass, currency, regions)
    4. RESOLVE_EFS - Emission factor resolution (EPA WARM, DEFRA, IPCC)
    5. CALCULATE - Treatment-pathway emissions calculation
    6. ALLOCATE - Allocation to treatment pathways based on regional splits
    7. AGGREGATE - Aggregation and summarization across products
    8. COMPLIANCE - Regulatory compliance checking (7 frameworks)
    9. PROVENANCE - Provenance metadata recording
    10. SEAL - Final sealing and verification

Data Models:
    - StageRecord: Frozen (immutable) record of a single pipeline stage
    - ProvenanceChain: Ordered chain of StageRecords with validation
    - BatchProvenance: Merkle-aggregated batch provenance for bulk operations

Standalone Hash Functions (15+):
    hash_input, hash_material, hash_treatment, hash_product_eol,
    hash_landfill_result, hash_incineration_result, hash_recycling_result,
    hash_composting_result, hash_avoided_emission, hash_calculation,
    hash_aggregation, hash_compliance, hash_dqi, hash_uncertainty,
    hash_circularity

Example:
    >>> builder = get_provenance_builder()
    >>> chain_id = builder.start_chain()
    >>> builder.record_stage(chain_id, ProvenanceStage.VALIDATE, raw, validated)
    >>> builder.record_stage(chain_id, ProvenanceStage.CALCULATE, calc_in, calc_out)
    >>> final_hash = builder.seal_chain(chain_id)
    >>> assert builder.validate_chain(chain_id)

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-012
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

AGENT_ID = "GL-MRV-S3-012"
AGENT_VERSION = "1.0.0"
HASH_ALGORITHM = "sha256"
ENCODING = "utf-8"


class ProvenanceStage(str, Enum):
    """
    Pipeline stages for end-of-life treatment provenance tracking.

    10 stages covering the complete calculation pipeline from input
    validation through final sealing.
    """

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
class StageRecord:
    """
    Single provenance stage record in the chain (frozen/immutable).

    Each record captures the input and output hashes for a pipeline stage,
    linked to the previous record via chain_hash to form an immutable chain.

    Attributes:
        record_id: Unique identifier for this record (UUID)
        stage: Pipeline stage name (from ProvenanceStage enum)
        timestamp: ISO 8601 UTC timestamp of recording
        input_hash: SHA-256 hash of stage input data
        output_hash: SHA-256 hash of stage output data
        chain_hash: SHA-256 hash linking to previous record
        previous_hash: Chain hash of previous record (empty for first)
        agent_id: Agent identifier (GL-MRV-S3-012)
        agent_version: Agent version (1.0.0)
        metadata: Additional context (optional)

    Example:
        >>> record = StageRecord(
        ...     record_id="abc-123",
        ...     stage="VALIDATE",
        ...     timestamp="2026-01-15T12:00:00+00:00",
        ...     input_hash="a1b2c3...",
        ...     output_hash="d4e5f6...",
        ...     chain_hash="g7h8i9...",
        ...     previous_hash=""
        ... )
    """

    record_id: str
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
        """Convert record to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert record to JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True)


@dataclass
class ProvenanceChain:
    """
    Complete provenance chain for a single calculation.

    Tracks all stage records in order with validation status and
    final hash when sealed.

    Attributes:
        chain_id: Unique identifier for this chain
        entries: Ordered list of StageRecords
        started_at: ISO 8601 UTC timestamp of chain start
        sealed_at: ISO 8601 UTC timestamp when sealed (or None)
        final_hash: Final SHA-256 hash when sealed (or None)
        chain_valid: Whether chain validation passed

    Example:
        >>> chain = ProvenanceChain(chain_id="chain-abc-123")
        >>> chain.chain_valid
        True
    """

    chain_id: str
    entries: List[StageRecord] = field(default_factory=list)
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    sealed_at: Optional[str] = None
    final_hash: Optional[str] = None
    chain_valid: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert chain to dictionary."""
        return {
            "chain_id": self.chain_id,
            "entries": [e.to_dict() for e in self.entries],
            "started_at": self.started_at,
            "sealed_at": self.sealed_at,
            "final_hash": self.final_hash,
            "chain_valid": self.chain_valid,
        }

    def to_json(self) -> str:
        """Convert chain to JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


@dataclass
class BatchProvenance:
    """
    Provenance tracking for batch calculations.

    Aggregates multiple individual chains into a single batch with
    Merkle-style hash aggregation.

    Attributes:
        batch_id: Unique identifier for this batch
        individual_chain_ids: List of chain IDs in this batch
        batch_started_at: ISO 8601 UTC timestamp
        batch_sealed_at: ISO 8601 UTC timestamp when sealed (or None)
        batch_hash: Merkle-style aggregate hash (or None)
        item_count: Number of items in batch

    Example:
        >>> batch = BatchProvenance(batch_id="batch-abc", item_count=100)
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

    Handles Decimal, datetime, Enum, and objects with to_dict().

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
        64-character lowercase hex SHA-256 hash
    """
    serialized = _serialize(data)
    return hashlib.sha256(serialized.encode(ENCODING)).hexdigest()


def _compute_chain_hash(
    previous_hash: str, stage: str, input_hash: str, output_hash: str
) -> str:
    """
    Compute chain hash linking to previous entry.

    Chain hash = SHA-256(previous_hash | stage | input_hash | output_hash)

    Args:
        previous_hash: Chain hash of previous entry (empty string for first)
        stage: Current stage name
        input_hash: Hash of input data
        output_hash: Hash of output data

    Returns:
        64-character lowercase hex SHA-256 chain hash
    """
    chain_data = f"{previous_hash}|{stage}|{input_hash}|{output_hash}"
    return hashlib.sha256(chain_data.encode(ENCODING)).hexdigest()


def _compute_merkle_root(hashes: List[str]) -> str:
    """
    Compute Merkle-style root hash from a list of hashes.

    Sorts hashes for determinism, then combines and hashes.

    Args:
        hashes: List of SHA-256 hashes

    Returns:
        64-character lowercase hex Merkle root hash
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
# PROVENANCE CHAIN BUILDER
# ============================================================================


class ProvenanceChainBuilder:
    """
    Thread-safe provenance chain builder.

    Manages multiple provenance chains, records pipeline stages, validates
    chain integrity, and seals chains for audit trail finalization.

    Uses RLock for thread-safe access to internal chain storage.

    Example:
        >>> builder = ProvenanceChainBuilder()
        >>> chain_id = builder.start_chain()
        >>> builder.record_stage(chain_id, "VALIDATE", input_data, output_data)
        >>> builder.record_stage(chain_id, "CALCULATE", calc_input, calc_output)
        >>> final_hash = builder.seal_chain(chain_id)
        >>> assert builder.validate_chain(chain_id)
    """

    def __init__(
        self,
        agent_id: str = AGENT_ID,
        agent_version: str = AGENT_VERSION,
    ):
        """
        Initialize provenance chain builder.

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
    ) -> StageRecord:
        """
        Record a pipeline stage in the provenance chain.

        Args:
            chain_id: Chain identifier
            stage: Pipeline stage name (string or ProvenanceStage enum)
            input_data: Input data for this stage
            output_data: Output data from this stage
            metadata: Optional metadata dict

        Returns:
            Created StageRecord

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
            stage_str = stage.value if isinstance(stage, ProvenanceStage) else stage

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

            # Create record
            record = StageRecord(
                record_id=str(uuid.uuid4()),
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

            chain.entries.append(record)
            return record

    def validate_chain(self, chain_id: str) -> bool:
        """
        Validate integrity of provenance chain.

        Checks:
        1. Each entry's chain_hash matches recomputed value
        2. Each entry's previous_hash matches previous entry's chain_hash
        3. First entry has empty previous_hash

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

            for i, entry in enumerate(chain.entries):
                # Check previous_hash linkage
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

    def seal_chain(self, chain_id: str) -> str:
        """
        Seal the provenance chain.

        Validates the chain, records the seal timestamp, and sets the
        final hash. No more entries can be added after sealing.

        Args:
            chain_id: Chain identifier

        Returns:
            Final hash (64-character hex)

        Raises:
            ValueError: If chain not found, already sealed, or validation fails
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
                final_hash = hashlib.sha256(chain_id.encode(ENCODING)).hexdigest()

            chain.sealed_at = datetime.now(timezone.utc).isoformat()
            chain.final_hash = final_hash
            return final_hash

    def build_chain(self, chain_id: str) -> ProvenanceChain:
        """
        Get the built provenance chain.

        Args:
            chain_id: Chain identifier

        Returns:
            ProvenanceChain instance

        Raises:
            ValueError: If chain not found
        """
        with self._lock:
            chain = self._chains.get(chain_id)
            if chain is None:
                raise ValueError(f"Chain {chain_id} not found")
            return chain

    def export_chain(self, chain_id: str, fmt: str = "json") -> str:
        """
        Export provenance chain as serialized string.

        Args:
            chain_id: Chain identifier
            fmt: Export format ("json")

        Returns:
            Serialized chain string

        Raises:
            ValueError: If chain not found or format unsupported
        """
        with self._lock:
            chain = self._chains.get(chain_id)
            if chain is None:
                raise ValueError(f"Chain {chain_id} not found")

            if fmt == "json":
                return chain.to_json()
            else:
                raise ValueError(f"Unsupported format: {fmt}")

    def verify_record(self, record: StageRecord) -> bool:
        """
        Verify a single stage record's chain hash.

        Recomputes the chain_hash and checks it matches.

        Args:
            record: StageRecord to verify

        Returns:
            True if record is valid, False otherwise
        """
        expected_hash = _compute_chain_hash(
            record.previous_hash,
            record.stage,
            record.input_hash,
            record.output_hash,
        )
        return record.chain_hash == expected_hash

    def get_stage_hash(
        self, chain_id: str, stage: Union[str, ProvenanceStage]
    ) -> Optional[str]:
        """
        Get output hash for a specific stage in the chain.

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

            stage_str = stage.value if isinstance(stage, ProvenanceStage) else stage
            for entry in chain.entries:
                if entry.stage == stage_str:
                    return entry.output_hash
            return None

    def compute_chain_hash(self, chain_id: str) -> str:
        """
        Compute the current chain hash (last entry's chain_hash).

        Args:
            chain_id: Chain identifier

        Returns:
            Current chain hash (64-character hex)

        Raises:
            ValueError: If chain not found or empty
        """
        with self._lock:
            chain = self._chains.get(chain_id)
            if chain is None:
                raise ValueError(f"Chain {chain_id} not found")
            if not chain.entries:
                return hashlib.sha256(chain_id.encode(ENCODING)).hexdigest()
            return chain.entries[-1].chain_hash

    def compute_merkle_root(self, chain_ids: List[str]) -> str:
        """
        Compute Merkle root hash across multiple chains.

        Args:
            chain_ids: List of chain identifiers

        Returns:
            Merkle root hash (64-character hex)

        Raises:
            ValueError: If any chain not found
        """
        with self._lock:
            hashes = []
            for cid in chain_ids:
                chain = self._chains.get(cid)
                if chain is None:
                    raise ValueError(f"Chain {cid} not found")
                if chain.final_hash:
                    hashes.append(chain.final_hash)
                elif chain.entries:
                    hashes.append(chain.entries[-1].chain_hash)
                else:
                    hashes.append(
                        hashlib.sha256(cid.encode(ENCODING)).hexdigest()
                    )
            return _compute_merkle_root(hashes)

    def get_chain_summary(self, chain_id: str) -> Dict[str, Any]:
        """
        Get summary of provenance chain.

        Args:
            chain_id: Chain identifier

        Returns:
            Summary dictionary with key metrics

        Raises:
            ValueError: If chain not found
        """
        with self._lock:
            chain = self._chains.get(chain_id)
            if chain is None:
                raise ValueError(f"Chain {chain_id} not found")

            stages = [e.stage for e in chain.entries]
            return {
                "chain_id": chain.chain_id,
                "agent_id": self.agent_id,
                "agent_version": self.agent_version,
                "started_at": chain.started_at,
                "sealed_at": chain.sealed_at,
                "final_hash": chain.final_hash,
                "chain_valid": chain.chain_valid,
                "entry_count": len(chain.entries),
                "stages": stages,
                "is_sealed": chain.sealed_at is not None,
            }

    def delete_chain(self, chain_id: str) -> bool:
        """
        Delete a provenance chain.

        Args:
            chain_id: Chain identifier

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if chain_id in self._chains:
                del self._chains[chain_id]
                return True
            return False

    def get_all_chains(self) -> List[str]:
        """
        Get list of all chain IDs.

        Returns:
            List of chain IDs
        """
        with self._lock:
            return list(self._chains.keys())

    def clear_all_chains(self) -> int:
        """
        Clear all chains (for testing).

        Returns:
            Number of chains cleared
        """
        with self._lock:
            count = len(self._chains)
            self._chains.clear()
            return count

    def reset(self) -> None:
        """
        Reset builder state (for testing).

        Clears all chains and resets internal state.
        """
        with self._lock:
            self._chains.clear()


# ============================================================================
# BATCH PROVENANCE BUILDER
# ============================================================================


class BatchProvenanceBuilder:
    """
    Batch provenance builder for Merkle-aggregated batch provenance.

    Manages multiple individual chains within a batch and computes
    a Merkle-style aggregate hash on seal.

    Example:
        >>> batch_builder = BatchProvenanceBuilder(chain_builder)
        >>> batch_id = batch_builder.start_batch(100)
        >>> for item in items:
        ...     chain_id = chain_builder.start_chain()
        ...     # ... process item ...
        ...     chain_builder.seal_chain(chain_id)
        ...     batch_builder.add_chain(batch_id, chain_id)
        >>> batch_hash = batch_builder.seal_batch(batch_id)
    """

    def __init__(self, chain_builder: ProvenanceChainBuilder):
        """
        Initialize batch provenance builder.

        Args:
            chain_builder: ProvenanceChainBuilder for individual chain access
        """
        self.chain_builder = chain_builder
        self._batches: Dict[str, BatchProvenance] = {}
        self._lock = threading.RLock()

    def start_batch(
        self, item_count: int = 0, batch_id: Optional[str] = None
    ) -> str:
        """
        Start a new batch.

        Args:
            item_count: Expected number of items
            batch_id: Optional batch ID (UUID generated if not provided)

        Returns:
            Batch ID

        Raises:
            ValueError: If batch_id already exists
        """
        with self._lock:
            if batch_id is None:
                batch_id = str(uuid.uuid4())

            if batch_id in self._batches:
                raise ValueError(f"Batch {batch_id} already exists")

            batch = BatchProvenance(batch_id=batch_id, item_count=item_count)
            self._batches[batch_id] = batch
            return batch_id

    def add_chain(self, batch_id: str, chain_id: str) -> None:
        """
        Add a chain to the batch.

        Args:
            batch_id: Batch identifier
            chain_id: Chain identifier to add

        Raises:
            ValueError: If batch not found or already sealed
        """
        with self._lock:
            batch = self._batches.get(batch_id)
            if batch is None:
                raise ValueError(f"Batch {batch_id} not found")
            if batch.batch_sealed_at is not None:
                raise ValueError(f"Batch {batch_id} already sealed")

            batch.individual_chain_ids.append(chain_id)

    def seal_batch(self, batch_id: str) -> str:
        """
        Seal the batch with Merkle-aggregated hash.

        Collects final hashes from all chains and computes a Merkle root.

        Args:
            batch_id: Batch identifier

        Returns:
            Batch Merkle hash (64-character hex)

        Raises:
            ValueError: If batch not found or already sealed
        """
        with self._lock:
            batch = self._batches.get(batch_id)
            if batch is None:
                raise ValueError(f"Batch {batch_id} not found")
            if batch.batch_sealed_at is not None:
                raise ValueError(f"Batch {batch_id} already sealed")

            # Collect final hashes from all chains
            chain_hashes: List[str] = []
            for chain_id in batch.individual_chain_ids:
                try:
                    chain = self.chain_builder.build_chain(chain_id)
                    if chain.final_hash:
                        chain_hashes.append(chain.final_hash)
                    elif chain.entries:
                        chain_hashes.append(chain.entries[-1].chain_hash)
                except ValueError:
                    # Chain not found -- skip
                    pass

            # Compute Merkle hash
            batch_hash = _compute_merkle_root(chain_hashes)
            batch.batch_hash = batch_hash
            batch.batch_sealed_at = datetime.now(timezone.utc).isoformat()
            return batch_hash

    def get_batch(self, batch_id: str) -> BatchProvenance:
        """
        Get batch provenance.

        Args:
            batch_id: Batch identifier

        Returns:
            BatchProvenance

        Raises:
            ValueError: If batch not found
        """
        with self._lock:
            batch = self._batches.get(batch_id)
            if batch is None:
                raise ValueError(f"Batch {batch_id} not found")
            return batch

    def get_batch_summary(self, batch_id: str) -> Dict[str, Any]:
        """
        Get summary of batch provenance.

        Args:
            batch_id: Batch identifier

        Returns:
            Summary dictionary

        Raises:
            ValueError: If batch not found
        """
        with self._lock:
            batch = self._batches.get(batch_id)
            if batch is None:
                raise ValueError(f"Batch {batch_id} not found")

            return {
                "batch_id": batch.batch_id,
                "batch_started_at": batch.batch_started_at,
                "batch_sealed_at": batch.batch_sealed_at,
                "batch_hash": batch.batch_hash,
                "item_count": batch.item_count,
                "chain_count": len(batch.individual_chain_ids),
                "individual_chain_ids": batch.individual_chain_ids,
                "is_sealed": batch.batch_sealed_at is not None,
            }


# ============================================================================
# STANDALONE HASH FUNCTIONS (15+)
# ============================================================================


def hash_input(request: Dict[str, Any]) -> str:
    """
    Hash complete calculation request input.

    Args:
        request: Calculation request dictionary

    Returns:
        64-character hex SHA-256 hash
    """
    return _compute_hash(request)


def hash_material(
    material_type: str,
    mass_kg: Union[Decimal, float],
    region: str,
) -> str:
    """
    Hash material composition data for a product.

    Args:
        material_type: Material type (e.g., "plastic_pet", "steel", "glass")
        mass_kg: Material mass in kilograms
        region: Region for waste treatment statistics

    Returns:
        64-character hex SHA-256 hash
    """
    data = {
        "material_type": material_type,
        "mass_kg": str(mass_kg),
        "region": region,
    }
    return _compute_hash(data)


def hash_treatment(
    treatment_method: str,
    treatment_share: Union[Decimal, float],
    region: str,
) -> str:
    """
    Hash treatment pathway allocation data.

    Args:
        treatment_method: Treatment method (landfill/incineration/recycling/etc)
        treatment_share: Share of waste going to this treatment (0.0-1.0)
        region: Region for treatment split statistics

    Returns:
        64-character hex SHA-256 hash
    """
    data = {
        "treatment_method": treatment_method,
        "treatment_share": str(treatment_share),
        "region": region,
    }
    return _compute_hash(data)


def hash_product_eol(
    product_id: str,
    product_category: str,
    total_mass_kg: Union[Decimal, float],
    materials: List[Dict[str, Any]],
) -> str:
    """
    Hash complete product end-of-life data.

    Args:
        product_id: Product identifier
        product_category: Product category
        total_mass_kg: Total product mass in kg
        materials: List of material composition dicts

    Returns:
        64-character hex SHA-256 hash
    """
    data = {
        "product_id": product_id,
        "product_category": product_category,
        "total_mass_kg": str(total_mass_kg),
        "materials": materials,
    }
    return _compute_hash(data)


def hash_landfill_result(
    mass_kg: Union[Decimal, float],
    doc: Union[Decimal, float],
    docf: Union[Decimal, float],
    mcf: Union[Decimal, float],
    ch4_kg: Union[Decimal, float],
    co2e_kg: Union[Decimal, float],
) -> str:
    """
    Hash landfill emissions calculation result.

    Args:
        mass_kg: Waste mass in kg
        doc: Degradable organic carbon fraction
        docf: DOC fraction dissimilated
        mcf: Methane correction factor
        ch4_kg: CH4 emissions in kg
        co2e_kg: Total CO2e emissions in kg

    Returns:
        64-character hex SHA-256 hash
    """
    data = {
        "mass_kg": str(mass_kg),
        "doc": str(doc),
        "docf": str(docf),
        "mcf": str(mcf),
        "ch4_kg": str(ch4_kg),
        "co2e_kg": str(co2e_kg),
        "formula": "mass * DOC * DOCf * MCF * F * (16/12) * GWP_CH4",
    }
    return _compute_hash(data)


def hash_incineration_result(
    mass_kg: Union[Decimal, float],
    fossil_fraction: Union[Decimal, float],
    oxidation_factor: Union[Decimal, float],
    co2_kg: Union[Decimal, float],
    energy_recovery_credit_kg: Union[Decimal, float],
    co2e_kg: Union[Decimal, float],
) -> str:
    """
    Hash incineration emissions calculation result.

    Args:
        mass_kg: Waste mass in kg
        fossil_fraction: Fossil carbon fraction
        oxidation_factor: Oxidation factor
        co2_kg: CO2 emissions in kg
        energy_recovery_credit_kg: Energy recovery credit in kg CO2e
        co2e_kg: Net CO2e emissions in kg

    Returns:
        64-character hex SHA-256 hash
    """
    data = {
        "mass_kg": str(mass_kg),
        "fossil_fraction": str(fossil_fraction),
        "oxidation_factor": str(oxidation_factor),
        "co2_kg": str(co2_kg),
        "energy_recovery_credit_kg": str(energy_recovery_credit_kg),
        "co2e_kg": str(co2e_kg),
        "formula": "mass * carbon_content * fossil_fraction * oxidation * (44/12)",
    }
    return _compute_hash(data)


def hash_recycling_result(
    mass_kg: Union[Decimal, float],
    process_emissions_kg: Union[Decimal, float],
    avoided_emissions_kg: Union[Decimal, float],
    net_co2e_kg: Union[Decimal, float],
) -> str:
    """
    Hash recycling emissions calculation result.

    Args:
        mass_kg: Waste mass in kg
        process_emissions_kg: Recycling process emissions in kg CO2e
        avoided_emissions_kg: Avoided emissions from substitution in kg CO2e
        net_co2e_kg: Net emissions in kg CO2e

    Returns:
        64-character hex SHA-256 hash
    """
    data = {
        "mass_kg": str(mass_kg),
        "process_emissions_kg": str(process_emissions_kg),
        "avoided_emissions_kg": str(avoided_emissions_kg),
        "net_co2e_kg": str(net_co2e_kg),
        "formula": "process_emissions - avoided_emissions",
    }
    return _compute_hash(data)


def hash_composting_result(
    mass_kg: Union[Decimal, float],
    ch4_kg: Union[Decimal, float],
    n2o_kg: Union[Decimal, float],
    co2e_kg: Union[Decimal, float],
) -> str:
    """
    Hash composting emissions calculation result.

    Args:
        mass_kg: Waste mass in kg
        ch4_kg: CH4 emissions in kg
        n2o_kg: N2O emissions in kg
        co2e_kg: Total CO2e emissions in kg

    Returns:
        64-character hex SHA-256 hash
    """
    data = {
        "mass_kg": str(mass_kg),
        "ch4_kg": str(ch4_kg),
        "n2o_kg": str(n2o_kg),
        "co2e_kg": str(co2e_kg),
        "formula": "mass * (CH4_EF * GWP_CH4 + N2O_EF * GWP_N2O)",
    }
    return _compute_hash(data)


def hash_avoided_emission(
    avoided_type: str,
    mass_kg: Union[Decimal, float],
    virgin_ef: Union[Decimal, float],
    recycled_ef: Union[Decimal, float],
    avoided_kg: Union[Decimal, float],
) -> str:
    """
    Hash avoided emission calculation.

    Args:
        avoided_type: Type (recycling/energy_recovery)
        mass_kg: Mass processed in kg
        virgin_ef: Virgin production EF (kg CO2e/kg)
        recycled_ef: Recycled production EF (kg CO2e/kg)
        avoided_kg: Avoided emissions in kg CO2e

    Returns:
        64-character hex SHA-256 hash
    """
    data = {
        "avoided_type": avoided_type,
        "mass_kg": str(mass_kg),
        "virgin_ef": str(virgin_ef),
        "recycled_ef": str(recycled_ef),
        "avoided_kg": str(avoided_kg),
        "formula": "mass * (virgin_ef - recycled_ef)",
    }
    return _compute_hash(data)


def hash_calculation(result: Dict[str, Any]) -> str:
    """
    Hash complete calculation result.

    Args:
        result: Calculation result dictionary

    Returns:
        64-character hex SHA-256 hash
    """
    return _compute_hash(result)


def hash_aggregation(
    by_treatment: Dict[str, Any],
    by_material: Dict[str, Any],
    by_product: Dict[str, Any],
    total_co2e_kg: Union[Decimal, float],
) -> str:
    """
    Hash aggregation result.

    Args:
        by_treatment: Emissions by treatment method
        by_material: Emissions by material type
        by_product: Emissions by product category
        total_co2e_kg: Total emissions in kg CO2e

    Returns:
        64-character hex SHA-256 hash
    """
    data = {
        "by_treatment": by_treatment,
        "by_material": by_material,
        "by_product": by_product,
        "total_co2e_kg": str(total_co2e_kg),
    }
    return _compute_hash(data)


def hash_compliance(
    framework: str,
    status: str,
    score: Union[Decimal, float],
    findings: Optional[List[str]] = None,
) -> str:
    """
    Hash compliance check result.

    Args:
        framework: Regulatory framework
        status: Compliance status
        score: Compliance score (0.0-1.0)
        findings: List of findings/issues (optional)

    Returns:
        64-character hex SHA-256 hash
    """
    data = {
        "framework": framework,
        "status": status,
        "score": str(score),
        "findings": findings or [],
    }
    return _compute_hash(data)


def hash_dqi(
    dqi_scores: Dict[str, Union[Decimal, float]],
    composite: Union[Decimal, float],
) -> str:
    """
    Hash data quality indicator assessment.

    Args:
        dqi_scores: DQI scores by dimension
        composite: Composite DQI score

    Returns:
        64-character hex SHA-256 hash
    """
    data = {
        "dqi_scores": {k: str(v) for k, v in dqi_scores.items()},
        "composite": str(composite),
    }
    return _compute_hash(data)


def hash_uncertainty(
    method: str,
    confidence: Union[Decimal, float],
    lower_bound: Union[Decimal, float],
    upper_bound: Union[Decimal, float],
) -> str:
    """
    Hash uncertainty quantification result.

    Args:
        method: Uncertainty method (MONTE_CARLO/BOOTSTRAP/etc)
        confidence: Confidence level (e.g., 0.95)
        lower_bound: Lower confidence bound
        upper_bound: Upper confidence bound

    Returns:
        64-character hex SHA-256 hash
    """
    data = {
        "method": method,
        "confidence": str(confidence),
        "lower_bound": str(lower_bound),
        "upper_bound": str(upper_bound),
    }
    return _compute_hash(data)


def hash_circularity(
    recycling_rate: Union[Decimal, float],
    diversion_rate: Union[Decimal, float],
    circularity_index: Union[Decimal, float],
    waste_hierarchy_score: Union[Decimal, float],
) -> str:
    """
    Hash circular economy metrics result.

    Args:
        recycling_rate: Recycling rate (0.0-1.0)
        diversion_rate: Waste diversion rate (0.0-1.0)
        circularity_index: Material Circularity Index (0.0-1.0)
        waste_hierarchy_score: Waste hierarchy score

    Returns:
        64-character hex SHA-256 hash
    """
    data = {
        "recycling_rate": str(recycling_rate),
        "diversion_rate": str(diversion_rate),
        "circularity_index": str(circularity_index),
        "waste_hierarchy_score": str(waste_hierarchy_score),
    }
    return _compute_hash(data)


# ============================================================================
# SINGLETON ACCESS
# ============================================================================


_builder_instance: Optional[ProvenanceChainBuilder] = None
_builder_lock = threading.RLock()


def get_provenance_builder() -> ProvenanceChainBuilder:
    """
    Get singleton ProvenanceChainBuilder instance.

    Thread-safe singleton pattern with double-checked locking.

    Returns:
        ProvenanceChainBuilder instance

    Example:
        >>> builder = get_provenance_builder()
        >>> chain_id = builder.start_chain()
    """
    global _builder_instance
    with _builder_lock:
        if _builder_instance is None:
            _builder_instance = ProvenanceChainBuilder()
        return _builder_instance


def reset_provenance_builder() -> None:
    """
    Reset singleton ProvenanceChainBuilder instance.

    Clears the singleton, forcing next get_provenance_builder() call
    to create a fresh instance. Primarily for testing.

    Example:
        >>> reset_provenance_builder()
        >>> builder = get_provenance_builder()  # Fresh instance
    """
    global _builder_instance
    with _builder_lock:
        _builder_instance = None


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def create_chain() -> Tuple[ProvenanceChainBuilder, str]:
    """
    Create a new provenance chain using singleton builder.

    Returns:
        Tuple of (builder, chain_id)

    Example:
        >>> builder, chain_id = create_chain()
        >>> builder.record_stage(chain_id, "VALIDATE", input_data, output_data)
    """
    builder = get_provenance_builder()
    chain_id = builder.start_chain()
    return builder, chain_id


def record_validation(
    chain_id: str,
    input_data: Any,
    validated_data: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> StageRecord:
    """
    Record VALIDATE stage using singleton builder.

    Args:
        chain_id: Chain identifier
        input_data: Raw input data
        validated_data: Validated data
        metadata: Optional metadata

    Returns:
        StageRecord
    """
    builder = get_provenance_builder()
    return builder.record_stage(
        chain_id, ProvenanceStage.VALIDATE, input_data, validated_data, metadata
    )


def record_classification(
    chain_id: str,
    input_data: Any,
    classified_data: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> StageRecord:
    """
    Record CLASSIFY stage using singleton builder.

    Args:
        chain_id: Chain identifier
        input_data: Input data
        classified_data: Classified data
        metadata: Optional metadata

    Returns:
        StageRecord
    """
    builder = get_provenance_builder()
    return builder.record_stage(
        chain_id, ProvenanceStage.CLASSIFY, input_data, classified_data, metadata
    )


def record_normalization(
    chain_id: str,
    input_data: Any,
    normalized_data: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> StageRecord:
    """
    Record NORMALIZE stage using singleton builder.

    Args:
        chain_id: Chain identifier
        input_data: Input data
        normalized_data: Normalized data
        metadata: Optional metadata

    Returns:
        StageRecord
    """
    builder = get_provenance_builder()
    return builder.record_stage(
        chain_id, ProvenanceStage.NORMALIZE, input_data, normalized_data, metadata
    )


def record_ef_resolution(
    chain_id: str,
    input_data: Any,
    resolved_data: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> StageRecord:
    """
    Record RESOLVE_EFS stage using singleton builder.

    Args:
        chain_id: Chain identifier
        input_data: Input data
        resolved_data: Resolved emission factors
        metadata: Optional metadata

    Returns:
        StageRecord
    """
    builder = get_provenance_builder()
    return builder.record_stage(
        chain_id, ProvenanceStage.RESOLVE_EFS, input_data, resolved_data, metadata
    )


def record_calculation(
    chain_id: str,
    input_data: Any,
    result_data: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> StageRecord:
    """
    Record CALCULATE stage using singleton builder.

    Args:
        chain_id: Chain identifier
        input_data: Calculation input
        result_data: Calculation result
        metadata: Optional metadata

    Returns:
        StageRecord
    """
    builder = get_provenance_builder()
    return builder.record_stage(
        chain_id, ProvenanceStage.CALCULATE, input_data, result_data, metadata
    )


def record_allocation(
    chain_id: str,
    input_data: Any,
    allocated_data: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> StageRecord:
    """
    Record ALLOCATE stage using singleton builder.

    Args:
        chain_id: Chain identifier
        input_data: Input data
        allocated_data: Allocated data
        metadata: Optional metadata

    Returns:
        StageRecord
    """
    builder = get_provenance_builder()
    return builder.record_stage(
        chain_id, ProvenanceStage.ALLOCATE, input_data, allocated_data, metadata
    )


def record_aggregation(
    chain_id: str,
    input_data: Any,
    aggregated_data: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> StageRecord:
    """
    Record AGGREGATE stage using singleton builder.

    Args:
        chain_id: Chain identifier
        input_data: Input data
        aggregated_data: Aggregated data
        metadata: Optional metadata

    Returns:
        StageRecord
    """
    builder = get_provenance_builder()
    return builder.record_stage(
        chain_id, ProvenanceStage.AGGREGATE, input_data, aggregated_data, metadata
    )


def record_compliance_check(
    chain_id: str,
    input_data: Any,
    compliance_data: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> StageRecord:
    """
    Record COMPLIANCE stage using singleton builder.

    Args:
        chain_id: Chain identifier
        input_data: Input data
        compliance_data: Compliance check results
        metadata: Optional metadata

    Returns:
        StageRecord
    """
    builder = get_provenance_builder()
    return builder.record_stage(
        chain_id, ProvenanceStage.COMPLIANCE, input_data, compliance_data, metadata
    )


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Constants
    "AGENT_ID",
    "AGENT_VERSION",
    "HASH_ALGORITHM",
    # Enum
    "ProvenanceStage",
    # Data models
    "StageRecord",
    "ProvenanceChain",
    "BatchProvenance",
    # Builder classes
    "ProvenanceChainBuilder",
    "BatchProvenanceBuilder",
    # Singleton access
    "get_provenance_builder",
    "reset_provenance_builder",
    # Convenience functions
    "create_chain",
    "record_validation",
    "record_classification",
    "record_normalization",
    "record_ef_resolution",
    "record_calculation",
    "record_allocation",
    "record_aggregation",
    "record_compliance_check",
    # Standalone hash functions
    "hash_input",
    "hash_material",
    "hash_treatment",
    "hash_product_eol",
    "hash_landfill_result",
    "hash_incineration_result",
    "hash_recycling_result",
    "hash_composting_result",
    "hash_avoided_emission",
    "hash_calculation",
    "hash_aggregation",
    "hash_compliance",
    "hash_dqi",
    "hash_uncertainty",
    "hash_circularity",
]
