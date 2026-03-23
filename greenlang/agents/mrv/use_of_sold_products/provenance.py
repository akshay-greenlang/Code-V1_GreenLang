# -*- coding: utf-8 -*-
"""
Use of Sold Products Provenance Tracking - AGENT-MRV-024

SHA-256 chain-hashed provenance for zero-hallucination audit trails.
10 pipeline stages tracked with deterministic hashing.
Agent: GL-MRV-S3-011

This module implements a complete provenance tracking system for use-of-sold-
products emissions calculations. Every step in the calculation pipeline is
recorded with SHA-256 hashes, creating an immutable audit trail that proves
no data was hallucinated or modified.

Pipeline Stages:
    1. VALIDATE - Input validation and data quality checks
    2. CLASSIFY - Product classification (direct/indirect/fuel)
    3. NORMALIZE - Unit normalization (energy, mass, volume, currency)
    4. RESOLVE_EFS - Emission factor resolution (IPCC, DEFRA, EPA, IEA)
    5. CALCULATE_DIRECT - Direct use-phase emissions calculation
    6. CALCULATE_INDIRECT - Indirect use-phase emissions calculation
    7. CALCULATE_FUELS - Fuels and feedstocks emissions calculation
    8. APPLY_LIFETIME - Product lifetime and degradation modeling
    9. COMPLIANCE - Regulatory compliance checking
    10. SEAL - Final sealing and verification

Example:
    >>> builder = get_provenance_builder()
    >>> chain_id = builder.start_chain()
    >>> builder.record_stage(chain_id, "VALIDATE", input_data, validated_data)
    >>> builder.record_stage(chain_id, "CALCULATE_DIRECT", calc_input, calc_output)
    >>> final_hash = builder.seal_chain(chain_id)
    >>> assert builder.validate_chain(chain_id)

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-011
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

AGENT_ID = "GL-MRV-S3-011"
AGENT_VERSION = "1.0.0"
HASH_ALGORITHM = "sha256"
ENCODING = "utf-8"


class ProvenanceStage(str, Enum):
    """Pipeline stages for provenance tracking."""

    VALIDATE = "VALIDATE"
    CLASSIFY = "CLASSIFY"
    NORMALIZE = "NORMALIZE"
    RESOLVE_EFS = "RESOLVE_EFS"
    CALCULATE_DIRECT = "CALCULATE_DIRECT"
    CALCULATE_INDIRECT = "CALCULATE_INDIRECT"
    CALCULATE_FUELS = "CALCULATE_FUELS"
    APPLY_LIFETIME = "APPLY_LIFETIME"
    COMPLIANCE = "COMPLIANCE"
    SEAL = "SEAL"


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass(frozen=True)
class StageRecord:
    """
    Single stage record in the provenance chain.

    This is a frozen (immutable) record of one stage in the calculation
    pipeline. The chain_hash links this record to the previous record,
    creating an immutable chain.

    Attributes:
        record_id: Unique identifier for this record (UUID)
        stage: Pipeline stage name
        timestamp: ISO 8601 UTC timestamp
        input_hash: SHA-256 hash of input data
        output_hash: SHA-256 hash of output data
        chain_hash: SHA-256 hash linking to previous record
        previous_hash: Chain hash of previous record (or empty for first)
        agent_id: Agent identifier (GL-MRV-S3-011)
        agent_version: Agent version (1.0.0)
        metadata: Additional context (optional)
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
    Complete provenance chain for a calculation.

    Tracks all stage records in order, with validation status and final hash
    when sealed.

    Attributes:
        chain_id: Unique identifier for this chain
        records: Ordered list of stage records
        started_at: ISO 8601 UTC timestamp of chain start
        sealed_at: ISO 8601 UTC timestamp when sealed (or None)
        final_hash: Final SHA-256 hash when sealed (or None)
        chain_valid: Whether chain validation passed
    """

    chain_id: str
    records: List[StageRecord] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    sealed_at: Optional[str] = None
    final_hash: Optional[str] = None
    chain_valid: bool = True

    @property
    def is_valid(self) -> bool:
        """Whether the chain is valid."""
        return self.chain_valid

    @property
    def root_hash(self) -> str:
        """Root hash of the chain (first record's chain_hash or empty)."""
        if self.records:
            return self.records[0].chain_hash
        return ""

    @property
    def is_sealed(self) -> bool:
        """Whether the chain has been sealed."""
        return self.sealed_at is not None

    @property
    def record_count(self) -> int:
        """Number of records in the chain."""
        return len(self.records)

    @property
    def stages(self) -> List[str]:
        """List of stage names in the chain."""
        return [r.stage for r in self.records]

    def to_dict(self) -> Dict[str, Any]:
        """Convert chain to dictionary."""
        return {
            "chain_id": self.chain_id,
            "records": [r.to_dict() for r in self.records],
            "started_at": self.started_at,
            "sealed_at": self.sealed_at,
            "final_hash": self.final_hash,
            "chain_valid": self.chain_valid,
            "is_valid": self.is_valid,
            "is_sealed": self.is_sealed,
            "root_hash": self.root_hash,
            "record_count": self.record_count,
            "stages": self.stages,
        }

    def to_json(self) -> str:
        """Convert chain to JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


@dataclass
class BatchProvenance:
    """
    Provenance tracking for batch calculations.

    Tracks multiple individual chains plus batch-level aggregation
    via Merkle hashing.

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
    batch_started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
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
    Compute chain hash linking to previous record.

    Chain hash = SHA-256(previous_hash + stage + input_hash + output_hash)

    Args:
        previous_hash: Chain hash of previous record (empty string for first)
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

    Used for batch aggregation. Sorts hashes for determinism before
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

    sorted_hashes = sorted(hashes)
    combined = "|".join(sorted_hashes)
    return hashlib.sha256(combined.encode(ENCODING)).hexdigest()


# ============================================================================
# PROVENANCE CHAIN BUILDER
# ============================================================================


class ProvenanceChainBuilder:
    """
    Main provenance tracking system for use-of-sold-products calculations.

    Manages multiple chains, records stages, validates integrity.
    Thread-safe with RLock.

    Example:
        >>> builder = ProvenanceChainBuilder()
        >>> chain_id = builder.start_chain()
        >>> builder.record_stage(chain_id, "VALIDATE", input_data, output_data)
        >>> builder.seal_chain(chain_id)
        >>> assert builder.validate_chain(chain_id)
    """

    def __init__(
        self, agent_id: str = AGENT_ID, agent_version: str = AGENT_VERSION
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
            stage: Pipeline stage name
            input_data: Input data for this stage
            output_data: Output data from this stage
            metadata: Optional metadata (e.g., method, source, EF details)

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

            stage_str = stage.value if isinstance(stage, ProvenanceStage) else stage

            input_hash = _compute_hash(input_data)
            output_hash = _compute_hash(output_data)

            previous_hash = ""
            if chain.records:
                previous_hash = chain.records[-1].chain_hash

            chain_hash = _compute_chain_hash(
                previous_hash, stage_str, input_hash, output_hash
            )

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

            chain.records.append(record)
            return record

    def validate_chain(self, chain_id: str) -> bool:
        """
        Validate integrity of provenance chain.

        Checks:
        1. Each record's chain_hash matches recomputed value
        2. Each record's previous_hash matches previous record's chain_hash
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

            if not chain.records:
                return True

            for i, record in enumerate(chain.records):
                if i == 0:
                    if record.previous_hash != "":
                        chain.chain_valid = False
                        return False
                else:
                    if record.previous_hash != chain.records[i - 1].chain_hash:
                        chain.chain_valid = False
                        return False

                expected_hash = _compute_chain_hash(
                    record.previous_hash,
                    record.stage,
                    record.input_hash,
                    record.output_hash,
                )
                if record.chain_hash != expected_hash:
                    chain.chain_valid = False
                    return False

            chain.chain_valid = True
            return True

    def seal_chain(self, chain_id: str) -> str:
        """
        Seal the provenance chain.

        Records final timestamp and hash. No more records can be added.

        Args:
            chain_id: Chain identifier

        Returns:
            Final hash (64-char lowercase hex)

        Raises:
            ValueError: If chain not found, already sealed, or fails validation
        """
        with self._lock:
            chain = self._chains.get(chain_id)
            if chain is None:
                raise ValueError(f"Chain {chain_id} not found")
            if chain.sealed_at is not None:
                raise ValueError(f"Chain {chain_id} already sealed")

            if not self.validate_chain(chain_id):
                raise ValueError(f"Chain {chain_id} failed validation")

            if chain.records:
                final_hash = chain.records[-1].chain_hash
            else:
                final_hash = hashlib.sha256(chain_id.encode(ENCODING)).hexdigest()

            chain.sealed_at = datetime.now(timezone.utc).isoformat()
            chain.final_hash = final_hash
            return final_hash

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

    def build_chain(
        self,
        stages: List[Tuple[Union[str, ProvenanceStage], Any, Any]],
        chain_id: Optional[str] = None,
        metadata_list: Optional[List[Optional[Dict[str, Any]]]] = None,
    ) -> str:
        """
        Build a complete chain from a list of stages in one call.

        This convenience method starts a chain, records all stages,
        and returns the chain ID (unsealed, ready for further stages
        or sealing).

        Args:
            stages: List of (stage, input_data, output_data) tuples
            chain_id: Optional chain ID
            metadata_list: Optional list of metadata dicts (one per stage)

        Returns:
            Chain ID

        Raises:
            ValueError: If stages list is empty
        """
        if not stages:
            raise ValueError("stages list cannot be empty")

        cid = self.start_chain(chain_id=chain_id)

        for i, (stage, input_data, output_data) in enumerate(stages):
            meta = None
            if metadata_list and i < len(metadata_list):
                meta = metadata_list[i]
            self.record_stage(cid, stage, input_data, output_data, metadata=meta)

        return cid

    def export_chain(
        self, chain_id: str, fmt: str = "json"
    ) -> str:
        """
        Export provenance chain.

        Args:
            chain_id: Chain identifier
            fmt: Export format ("json" or "dict")

        Returns:
            Serialized chain

        Raises:
            ValueError: If chain not found or format unsupported
        """
        with self._lock:
            chain = self._chains.get(chain_id)
            if chain is None:
                raise ValueError(f"Chain {chain_id} not found")

            if fmt == "json":
                return chain.to_json()
            elif fmt == "dict":
                return json.dumps(chain.to_dict(), indent=2, sort_keys=True)
            else:
                raise ValueError(f"Unsupported format: {fmt}")

    def verify_record(self, record: StageRecord) -> bool:
        """
        Verify a single stage record.

        Recomputes chain_hash and checks it matches.

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

            stage_str = stage.value if isinstance(stage, ProvenanceStage) else stage
            for record in chain.records:
                if record.stage == stage_str:
                    return record.output_hash
            return None

    def compute_chain_hash(self, chain_id: str) -> str:
        """
        Compute the current chain hash without sealing.

        Args:
            chain_id: Chain identifier

        Returns:
            Current chain hash (last record's chain_hash or empty-string hash)

        Raises:
            ValueError: If chain not found
        """
        with self._lock:
            chain = self._chains.get(chain_id)
            if chain is None:
                raise ValueError(f"Chain {chain_id} not found")

            if chain.records:
                return chain.records[-1].chain_hash
            return hashlib.sha256(chain_id.encode(ENCODING)).hexdigest()

    def compute_merkle_root(self, chain_ids: List[str]) -> str:
        """
        Compute Merkle root hash over multiple chains.

        Collects final hashes from sealed chains and computes
        a Merkle-style aggregate hash.

        Args:
            chain_ids: List of chain identifiers

        Returns:
            Merkle root hash (64-char lowercase hex)

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
                elif chain.records:
                    hashes.append(chain.records[-1].chain_hash)
            return _merkle_hash(hashes)

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

            return {
                "chain_id": chain.chain_id,
                "agent_id": self.agent_id,
                "agent_version": self.agent_version,
                "started_at": chain.started_at,
                "sealed_at": chain.sealed_at,
                "final_hash": chain.final_hash,
                "chain_valid": chain.chain_valid,
                "record_count": chain.record_count,
                "stages": chain.stages,
                "is_sealed": chain.is_sealed,
                "root_hash": chain.root_hash,
            }

    def reset(self) -> None:
        """
        Clear all chains tracked by this builder instance.

        Useful for testing. Removes all chain state.
        """
        with self._lock:
            self._chains.clear()

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


# ============================================================================
# BATCH PROVENANCE BUILDER
# ============================================================================


class BatchProvenanceBuilder:
    """
    Provenance tracking for batch calculations.

    Manages multiple individual chains plus batch-level aggregation
    via Merkle hashing.

    Example:
        >>> batch_builder = BatchProvenanceBuilder(builder)
        >>> batch_id = batch_builder.start_batch(100)
        >>> for item in items:
        ...     chain_id = builder.start_chain()
        ...     # ... process item ...
        ...     batch_builder.add_chain(batch_id, chain_id)
        >>> batch_hash = batch_builder.seal_batch(batch_id)
    """

    def __init__(self, builder: ProvenanceChainBuilder):
        """
        Initialize batch provenance builder.

        Args:
            builder: ProvenanceChainBuilder instance for individual chains
        """
        self.builder = builder
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
        Seal the batch.

        Computes Merkle-style aggregate hash of all chain hashes.

        Args:
            batch_id: Batch identifier

        Returns:
            Batch hash (64-char lowercase hex)

        Raises:
            ValueError: If batch not found or already sealed
        """
        with self._lock:
            batch = self._batches.get(batch_id)
            if batch is None:
                raise ValueError(f"Batch {batch_id} not found")
            if batch.batch_sealed_at is not None:
                raise ValueError(f"Batch {batch_id} already sealed")

            chain_hashes = []
            for chain_id in batch.individual_chain_ids:
                try:
                    chain = self.builder.get_chain(chain_id)
                    if chain.final_hash:
                        chain_hashes.append(chain.final_hash)
                except ValueError:
                    pass

            batch_hash = _merkle_hash(chain_hashes)
            batch.batch_hash = batch_hash
            batch.batch_sealed_at = datetime.now(timezone.utc).isoformat()
            return batch_hash

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


# ============================================================================
# STANDALONE HASH FUNCTIONS
# ============================================================================


def hash_input(request: Dict[str, Any]) -> str:
    """
    Hash complete calculation request input.

    Args:
        request: Calculation request dictionary

    Returns:
        SHA-256 hash (64-char lowercase hex)
    """
    return _compute_hash(request)


def hash_emission_factor(
    ef_type: str, ef_value: Union[Decimal, float], source: str
) -> str:
    """
    Hash emission factor.

    Args:
        ef_type: Emission factor type (e.g., "natural_gas_combustion",
                 "grid_us_average", "r410a_gwp")
        ef_value: Emission factor value (kgCO2e per unit)
        source: Data source (e.g., "IPCC", "DEFRA", "EPA", "IEA")

    Returns:
        SHA-256 hash (64-char lowercase hex)
    """
    data = {"type": ef_type, "value": str(ef_value), "source": source}
    return _compute_hash(data)


def hash_product(
    product_id: str,
    product_category: str,
    units_sold: int,
    product_name: Optional[str] = None,
) -> str:
    """
    Hash product data.

    Args:
        product_id: Product identifier
        product_category: Product category (fuel_consuming, electricity_consuming, etc.)
        units_sold: Number of units sold
        product_name: Optional product name

    Returns:
        SHA-256 hash (64-char lowercase hex)
    """
    data = {
        "product_id": product_id,
        "product_category": product_category,
        "units_sold": units_sold,
        "product_name": product_name or "",
    }
    return _compute_hash(data)


def hash_direct_emission(
    product_id: str,
    emission_subtype: str,
    ef_value: Union[Decimal, float],
    activity_value: Union[Decimal, float],
    co2e_kg: Union[Decimal, float],
) -> str:
    """
    Hash direct use-phase emission calculation.

    Args:
        product_id: Product identifier
        emission_subtype: Direct emission sub-type (fuel/refrigerant/chemical)
        ef_value: Emission factor value
        activity_value: Activity data value
        co2e_kg: Total emissions in kgCO2e

    Returns:
        SHA-256 hash (64-char lowercase hex)
    """
    data = {
        "product_id": product_id,
        "emission_subtype": emission_subtype,
        "ef_value": str(ef_value),
        "activity_value": str(activity_value),
        "co2e_kg": str(co2e_kg),
        "formula": "ef_value * activity_value",
    }
    return _compute_hash(data)


def hash_indirect_emission(
    product_id: str,
    energy_type: str,
    energy_consumption_kwh: Union[Decimal, float],
    grid_ef: Union[Decimal, float],
    co2e_kg: Union[Decimal, float],
) -> str:
    """
    Hash indirect use-phase emission calculation.

    Args:
        product_id: Product identifier
        energy_type: Energy type (electricity/heating/steam)
        energy_consumption_kwh: Energy consumption in kWh
        grid_ef: Grid emission factor (kgCO2e/kWh)
        co2e_kg: Total emissions in kgCO2e

    Returns:
        SHA-256 hash (64-char lowercase hex)
    """
    data = {
        "product_id": product_id,
        "energy_type": energy_type,
        "energy_consumption_kwh": str(energy_consumption_kwh),
        "grid_ef": str(grid_ef),
        "co2e_kg": str(co2e_kg),
        "formula": "energy_consumption_kwh * grid_ef",
    }
    return _compute_hash(data)


def hash_fuel_sale(
    fuel_type: str,
    volume_sold: Union[Decimal, float],
    volume_unit: str,
    ef_value: Union[Decimal, float],
    co2e_kg: Union[Decimal, float],
) -> str:
    """
    Hash fuel sale emission calculation.

    Args:
        fuel_type: Fuel type (gasoline/diesel/natural_gas/coal/etc.)
        volume_sold: Volume of fuel sold
        volume_unit: Volume unit (litres/m3/tonnes/gallons)
        ef_value: Fuel combustion emission factor
        co2e_kg: Total emissions in kgCO2e

    Returns:
        SHA-256 hash (64-char lowercase hex)
    """
    data = {
        "fuel_type": fuel_type,
        "volume_sold": str(volume_sold),
        "volume_unit": volume_unit,
        "ef_value": str(ef_value),
        "co2e_kg": str(co2e_kg),
        "formula": "volume_sold * ef_value",
    }
    return _compute_hash(data)


def hash_calculation(
    method: str,
    inputs: Dict[str, Any],
    result_co2e_kg: Union[Decimal, float],
) -> str:
    """
    Hash a generic calculation result.

    Args:
        method: Calculation method name
        inputs: Input parameters dictionary
        result_co2e_kg: Result emissions in kgCO2e

    Returns:
        SHA-256 hash (64-char lowercase hex)
    """
    data = {
        "method": method,
        "inputs": inputs,
        "result_co2e_kg": str(result_co2e_kg),
    }
    return _compute_hash(data)


def hash_lifetime(
    product_id: str,
    lifetime_years: Union[Decimal, float, int],
    survival_model: str,
    degradation_rate: Union[Decimal, float],
) -> str:
    """
    Hash product lifetime modeling parameters.

    Args:
        product_id: Product identifier
        lifetime_years: Product lifetime in years
        survival_model: Survival curve model (WEIBULL/EXPONENTIAL/LINEAR)
        degradation_rate: Annual degradation rate

    Returns:
        SHA-256 hash (64-char lowercase hex)
    """
    data = {
        "product_id": product_id,
        "lifetime_years": str(lifetime_years),
        "survival_model": survival_model,
        "degradation_rate": str(degradation_rate),
    }
    return _compute_hash(data)


def hash_degradation(
    product_id: str,
    year: int,
    efficiency_factor: Union[Decimal, float],
    model: str,
    annual_rate: Union[Decimal, float],
) -> str:
    """
    Hash degradation curve data point.

    Args:
        product_id: Product identifier
        year: Year in product lifetime
        efficiency_factor: Remaining efficiency factor (0-1)
        model: Degradation model (LINEAR/EXPONENTIAL/STEP)
        annual_rate: Annual degradation rate

    Returns:
        SHA-256 hash (64-char lowercase hex)
    """
    data = {
        "product_id": product_id,
        "year": year,
        "efficiency_factor": str(efficiency_factor),
        "model": model,
        "annual_rate": str(annual_rate),
    }
    return _compute_hash(data)


def hash_allocation(
    product_id: str,
    allocation_method: str,
    allocation_factor: Union[Decimal, float],
    allocated_co2e_kg: Union[Decimal, float],
) -> str:
    """
    Hash emission allocation result.

    Args:
        product_id: Product identifier
        allocation_method: Allocation method name
        allocation_factor: Allocation factor (0-1)
        allocated_co2e_kg: Allocated emissions in kgCO2e

    Returns:
        SHA-256 hash (64-char lowercase hex)
    """
    data = {
        "product_id": product_id,
        "allocation_method": allocation_method,
        "allocation_factor": str(allocation_factor),
        "allocated_co2e_kg": str(allocated_co2e_kg),
    }
    return _compute_hash(data)


def hash_aggregation(
    product_ids: List[str],
    total_co2e_kg: Union[Decimal, float],
    product_count: int,
    aggregation_method: str = "sum",
) -> str:
    """
    Hash aggregation result across multiple products.

    Args:
        product_ids: List of product identifiers included
        total_co2e_kg: Total aggregated emissions in kgCO2e
        product_count: Number of products aggregated
        aggregation_method: Aggregation method (sum/weighted_sum/average)

    Returns:
        SHA-256 hash (64-char lowercase hex)
    """
    data = {
        "product_ids": sorted(product_ids),
        "total_co2e_kg": str(total_co2e_kg),
        "product_count": product_count,
        "aggregation_method": aggregation_method,
    }
    return _compute_hash(data)


def hash_compliance(
    framework: str,
    status: str,
    findings: List[Dict[str, Any]],
) -> str:
    """
    Hash compliance check result.

    Args:
        framework: Compliance framework name
        status: Compliance status (compliant/non_compliant/warning)
        findings: List of compliance findings

    Returns:
        SHA-256 hash (64-char lowercase hex)
    """
    data = {
        "framework": framework,
        "status": status,
        "findings": findings,
    }
    return _compute_hash(data)


def hash_dqi(
    technological: Union[Decimal, float],
    temporal: Union[Decimal, float],
    geographical: Union[Decimal, float],
    completeness: Union[Decimal, float],
    reliability: Union[Decimal, float],
    overall: Union[Decimal, float],
) -> str:
    """
    Hash data quality indicator scores.

    Args:
        technological: Technological representativeness score (1-5)
        temporal: Temporal representativeness score (1-5)
        geographical: Geographical representativeness score (1-5)
        completeness: Data completeness score (1-5)
        reliability: Data reliability score (1-5)
        overall: Overall weighted DQI score (1-5)

    Returns:
        SHA-256 hash (64-char lowercase hex)
    """
    data = {
        "technological": str(technological),
        "temporal": str(temporal),
        "geographical": str(geographical),
        "completeness": str(completeness),
        "reliability": str(reliability),
        "overall": str(overall),
    }
    return _compute_hash(data)


def hash_uncertainty(
    method: str,
    lower_bound: Union[Decimal, float],
    upper_bound: Union[Decimal, float],
    confidence_level: Union[Decimal, float],
    central_value: Union[Decimal, float],
) -> str:
    """
    Hash uncertainty quantification result.

    Args:
        method: Uncertainty method (MONTE_CARLO/BOOTSTRAP/IPCC_DEFAULT)
        lower_bound: Lower bound of confidence interval in kgCO2e
        upper_bound: Upper bound of confidence interval in kgCO2e
        confidence_level: Confidence level (e.g., 0.95)
        central_value: Central estimate in kgCO2e

    Returns:
        SHA-256 hash (64-char lowercase hex)
    """
    data = {
        "method": method,
        "lower_bound": str(lower_bound),
        "upper_bound": str(upper_bound),
        "confidence_level": str(confidence_level),
        "central_value": str(central_value),
    }
    return _compute_hash(data)


def hash_stage(
    stage_name: str,
    input_data: Any,
    output_data: Any,
) -> str:
    """
    Hash a complete pipeline stage (input + output combined).

    Args:
        stage_name: Pipeline stage name
        input_data: Input data for the stage
        output_data: Output data from the stage

    Returns:
        SHA-256 hash (64-char lowercase hex)
    """
    data = {
        "stage": stage_name,
        "input_hash": _compute_hash(input_data),
        "output_hash": _compute_hash(output_data),
    }
    return _compute_hash(data)


# ============================================================================
# MODULE-LEVEL SINGLETON
# ============================================================================

_builder_instance: Optional[ProvenanceChainBuilder] = None
_builder_lock = threading.RLock()


def get_provenance_builder() -> ProvenanceChainBuilder:
    """
    Get the singleton ProvenanceChainBuilder instance.

    Thread-safe accessor for the global provenance builder. Prefer this
    function over direct instantiation for consistency across the
    use-of-sold-products agent codebase.

    Returns:
        ProvenanceChainBuilder singleton instance

    Example:
        >>> from greenlang.agents.mrv.use_of_sold_products.provenance import get_provenance_builder
        >>> builder = get_provenance_builder()
        >>> chain_id = builder.start_chain()
        >>> builder.record_stage(chain_id, "VALIDATE", input_data, output_data)
        >>> builder.seal_chain(chain_id)
    """
    global _builder_instance

    if _builder_instance is None:
        with _builder_lock:
            if _builder_instance is None:
                _builder_instance = ProvenanceChainBuilder()

    return _builder_instance


def reset_provenance_builder() -> None:
    """
    Reset the singleton provenance builder for testing purposes.

    Clears the singleton instance so a fresh one is created on next access.
    Should only be called in test teardown.

    Example:
        >>> from greenlang.agents.mrv.use_of_sold_products.provenance import reset_provenance_builder
        >>> reset_provenance_builder()
    """
    global _builder_instance

    with _builder_lock:
        if _builder_instance is not None:
            _builder_instance.reset()
        _builder_instance = None


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
    "StageRecord",
    "ProvenanceChain",
    "BatchProvenance",
    # Builder classes
    "ProvenanceChainBuilder",
    "BatchProvenanceBuilder",
    # Standalone hash functions
    "hash_input",
    "hash_emission_factor",
    "hash_product",
    "hash_direct_emission",
    "hash_indirect_emission",
    "hash_fuel_sale",
    "hash_calculation",
    "hash_lifetime",
    "hash_degradation",
    "hash_allocation",
    "hash_aggregation",
    "hash_compliance",
    "hash_dqi",
    "hash_uncertainty",
    "hash_stage",
    # Module-level accessors
    "get_provenance_builder",
    "reset_provenance_builder",
]
