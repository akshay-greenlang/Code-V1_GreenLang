# -*- coding: utf-8 -*-
"""
Processing of Sold Products Provenance Tracking - AGENT-MRV-023

SHA-256 chain-hashed provenance for zero-hallucination audit trails.
10 pipeline stages tracked with deterministic hashing.
Agent: GL-MRV-S3-010

This module implements a complete provenance tracking system for processing
of sold products emissions calculations. Every step in the calculation
pipeline is recorded with SHA-256 hashes, creating an immutable audit
trail that proves no data was hallucinated or modified.

The provenance system supports:
- 10-stage pipeline tracking (VALIDATE through SEAL)
- SHA-256 deterministic hashing of all inputs and outputs
- Chain hashing linking each stage to its predecessor
- Merkle tree aggregation for batch provenance
- Domain-specific hash functions for intermediate products, emission factors,
  processing chains, allocation, aggregation, compliance, DQI, and uncertainty
- Chain validation to detect any tampering or data corruption
- JSON-serializable export for audit trail persistence

Pipeline Stages:
    1. VALIDATE     - Input validation and data quality checks
    2. CLASSIFY     - Product classification (category, processing type)
    3. NORMALIZE    - Unit normalization (mass, energy, currency)
    4. RESOLVE_EFS  - Emission factor resolution (DEFRA, EPA, ecoinvent, IEA)
    5. CALCULATE    - Processing emissions calculation (site-specific/avg/spend)
    6. ALLOCATE     - Allocation to products (mass, revenue, units, equal)
    7. AGGREGATE    - Aggregation and summarization
    8. COMPLIANCE   - Regulatory compliance checking (7 frameworks)
    9. PROVENANCE   - Provenance hash computation
    10. SEAL        - Final sealing and verification

GHG Protocol Scope 3 Category 10 covers emissions from downstream
processing of intermediate products sold by the reporting company.
This provenance module ensures every calculation step is auditable
and reproducible, meeting zero-hallucination requirements.

Example:
    >>> builder = get_provenance_builder()
    >>> chain_id = builder.start_chain()
    >>> builder.record_stage(chain_id, ProvenanceStage.VALIDATE, input_data, validated_data)
    >>> builder.record_stage(chain_id, ProvenanceStage.CALCULATE, calc_input, calc_output)
    >>> chain = builder.get_chain(chain_id)
    >>> sealed = builder.seal_chain(chain)
    >>> assert builder.validate_chain(chain)

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-010
"""

import hashlib
import json
import logging
import math
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

AGENT_ID = "GL-MRV-S3-010"
AGENT_VERSION = "1.0.0"
HASH_ALGORITHM = "sha256"
ENCODING = "utf-8"


# ============================================================================
# ENUMERATIONS
# ============================================================================


class ProvenanceStage(str, Enum):
    """
    Pipeline stages for provenance tracking.

    Each stage represents a distinct phase in the processing of sold products
    emissions calculation pipeline. Stages must be recorded in order, and
    each stage's hash links to the previous stage's hash, forming an
    immutable chain.

    Stages:
        VALIDATE:    Input validation and data quality checks
        CLASSIFY:    Product classification (category, processing type)
        NORMALIZE:   Unit normalization (mass, energy, currency)
        RESOLVE_EFS: Emission factor resolution (DEFRA, EPA, ecoinvent, IEA)
        CALCULATE:   Processing emissions calculation
        ALLOCATE:    Allocation to products (mass, revenue, units, equal)
        AGGREGATE:   Aggregation and summarization
        COMPLIANCE:  Regulatory compliance checking (7 frameworks)
        PROVENANCE:  Provenance hash computation
        SEAL:        Final sealing and verification
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


# Canonical stage ordering for validation
_STAGE_ORDER: Dict[str, int] = {
    stage.value: idx for idx, stage in enumerate(ProvenanceStage)
}


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass(frozen=True)
class StageRecord:
    """
    Single provenance record for one pipeline stage.

    Frozen (immutable) record of one stage in the calculation pipeline.
    The output of one stage becomes the input of the next, and the
    chain_hash links this record to the previous record.

    Attributes:
        stage: Pipeline stage enum value
        input_hash: SHA-256 hash of input data (64-char lowercase hex)
        output_hash: SHA-256 hash of output data (64-char lowercase hex)
        timestamp: ISO 8601 UTC timestamp of when this stage was recorded
        metadata: Additional context (optional, e.g., method used, EF source)
    """

    stage: ProvenanceStage
    input_hash: str
    output_hash: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert stage record to a JSON-serializable dictionary.

        Returns:
            Dictionary representation with stage as string value.
        """
        return {
            "stage": self.stage.value if isinstance(self.stage, ProvenanceStage) else self.stage,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
        }

    def to_json(self) -> str:
        """
        Convert stage record to deterministic JSON string.

        Returns:
            JSON string with sorted keys.
        """
        return json.dumps(self.to_dict(), sort_keys=True, default=str)


@dataclass
class ProvenanceChain:
    """
    Complete provenance chain for a processing of sold products calculation.

    Tracks all stage records in order with chain-level hashes, Merkle root,
    and seal status. Once sealed, no further stages can be added.

    Attributes:
        chain_id: Unique identifier for this chain (UUID)
        stages: Ordered list of StageRecord objects
        chain_hash: Sequential chain hash of all stages (64-char lowercase hex)
        merkle_root: Merkle tree root hash of all stage hashes (64-char lowercase hex)
        sealed_at: ISO 8601 UTC timestamp when sealed (None if unsealed)
        agent_id: Agent identifier (default: GL-MRV-S3-010)
    """

    chain_id: str
    stages: List[StageRecord] = field(default_factory=list)
    chain_hash: str = ""
    merkle_root: str = ""
    sealed_at: str = ""
    agent_id: str = AGENT_ID

    @property
    def is_sealed(self) -> bool:
        """Whether the chain has been sealed."""
        return bool(self.sealed_at)

    @property
    def stage_count(self) -> int:
        """Number of stages recorded in this chain."""
        return len(self.stages)

    @property
    def stage_names(self) -> List[str]:
        """List of stage names in recorded order."""
        return [
            s.stage.value if isinstance(s.stage, ProvenanceStage) else str(s.stage)
            for s in self.stages
        ]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert chain to a JSON-serializable dictionary.

        Returns:
            Dictionary representation with all stages and hashes.
        """
        return {
            "chain_id": self.chain_id,
            "stages": [s.to_dict() for s in self.stages],
            "chain_hash": self.chain_hash,
            "merkle_root": self.merkle_root,
            "sealed_at": self.sealed_at,
            "agent_id": self.agent_id,
            "is_sealed": self.is_sealed,
            "stage_count": self.stage_count,
            "stage_names": self.stage_names,
        }

    def to_json(self) -> str:
        """
        Convert chain to JSON string.

        Returns:
            JSON string with sorted keys and indentation.
        """
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str)


@dataclass
class BatchProvenance:
    """
    Provenance tracking for batch calculations with Merkle tree support.

    Aggregates multiple individual ProvenanceChain instances into a single
    batch with a Merkle tree root hash for efficient verification.

    Attributes:
        batch_id: Unique identifier for this batch
        individual_chain_ids: List of chain IDs in this batch
        batch_started_at: ISO 8601 UTC timestamp
        batch_sealed_at: ISO 8601 UTC timestamp when sealed
        batch_hash: Merkle root hash of all chain hashes
        item_count: Number of items in batch
        leaf_hashes: Leaf-level hashes for Merkle tree
        tree_levels: Merkle tree levels (list of lists)
        root_hash: Merkle root hash
    """

    batch_id: str
    individual_chain_ids: List[str] = field(default_factory=list)
    batch_started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    batch_sealed_at: Optional[str] = None
    batch_hash: Optional[str] = None
    item_count: int = 0
    leaf_hashes: List[str] = field(default_factory=list)
    tree_levels: List[List[str]] = field(default_factory=list)
    root_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert batch provenance to dictionary."""
        return {
            "batch_id": self.batch_id,
            "individual_chain_ids": self.individual_chain_ids,
            "batch_started_at": self.batch_started_at,
            "batch_sealed_at": self.batch_sealed_at,
            "batch_hash": self.batch_hash,
            "item_count": self.item_count,
            "leaf_hashes": self.leaf_hashes,
            "tree_levels_count": len(self.tree_levels),
            "root_hash": self.root_hash,
        }


# ============================================================================
# HASH UTILITIES
# ============================================================================


def _serialize(obj: Any) -> str:
    """
    Serialize object to deterministic JSON string.

    Converts Decimal to string, datetime to ISO format, Enum to value,
    sorts keys, handles nested dicts/lists and frozen models.

    Args:
        obj: Object to serialize (any JSON-serializable type)

    Returns:
        Deterministic JSON string with sorted keys
    """

    def default_handler(o: Any) -> Any:
        """Handle non-JSON-serializable types."""
        if isinstance(o, Decimal):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, Enum):
            return o.value
        if isinstance(o, (set, frozenset)):
            return sorted(str(x) for x in o)
        if hasattr(o, "to_dict"):
            return o.to_dict()
        if hasattr(o, "__dict__"):
            return o.__dict__
        return str(o)

    return json.dumps(obj, sort_keys=True, default=default_handler)


def _compute_hash(data: Any) -> str:
    """
    Compute SHA-256 hash of data.

    Serializes the data to a deterministic JSON string and computes
    the SHA-256 hash. The result is always a 64-character lowercase
    hexadecimal string.

    Args:
        data: Data to hash (any JSON-serializable type)

    Returns:
        64-character lowercase hex SHA-256 hash
    """
    serialized = _serialize(data)
    return hashlib.sha256(serialized.encode(ENCODING)).hexdigest()


def _compute_chain_link(
    previous_hash: str,
    stage: str,
    input_hash: str,
    output_hash: str,
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


def _merkle_hash_pair(left: str, right: str) -> str:
    """
    Compute Merkle hash of two sibling nodes.

    Args:
        left: Left child hash (64-char hex)
        right: Right child hash (64-char hex)

    Returns:
        64-character lowercase hex SHA-256 parent hash
    """
    combined = f"{left}|{right}"
    return hashlib.sha256(combined.encode(ENCODING)).hexdigest()


def _build_merkle_tree(leaf_hashes: List[str]) -> Tuple[List[List[str]], str]:
    """
    Build a full Merkle tree from leaf hashes.

    If the number of leaves at any level is odd, the last leaf is
    duplicated to form a complete binary tree.

    Args:
        leaf_hashes: List of leaf-level SHA-256 hashes

    Returns:
        Tuple of (tree_levels, root_hash) where tree_levels[0] is the
        leaf level and tree_levels[-1] contains the root.
    """
    if not leaf_hashes:
        empty_hash = hashlib.sha256(b"").hexdigest()
        return [[empty_hash]], empty_hash

    if len(leaf_hashes) == 1:
        return [leaf_hashes], leaf_hashes[0]

    levels: List[List[str]] = [list(leaf_hashes)]
    current_level = list(leaf_hashes)

    while len(current_level) > 1:
        next_level: List[str] = []
        # Duplicate last element if odd count
        if len(current_level) % 2 != 0:
            current_level.append(current_level[-1])

        for i in range(0, len(current_level), 2):
            parent = _merkle_hash_pair(current_level[i], current_level[i + 1])
            next_level.append(parent)

        levels.append(next_level)
        current_level = next_level

    root_hash = current_level[0]
    return levels, root_hash


def _verify_merkle_proof(
    leaf_hash: str,
    proof: List[Tuple[str, str]],
    root_hash: str,
) -> bool:
    """
    Verify a Merkle inclusion proof.

    Args:
        leaf_hash: The leaf hash to verify
        proof: List of (sibling_hash, side) tuples where side is 'left' or 'right'
        root_hash: Expected Merkle root hash

    Returns:
        True if the proof is valid
    """
    current = leaf_hash
    for sibling_hash, side in proof:
        if side == "left":
            current = _merkle_hash_pair(sibling_hash, current)
        else:
            current = _merkle_hash_pair(current, sibling_hash)
    return current == root_hash


# ============================================================================
# PROVENANCE CHAIN BUILDER -- Thread-safe Singleton
# ============================================================================


class ProvenanceChainBuilder:
    """
    Thread-safe singleton for building and managing provenance chains
    for the Processing of Sold Products agent (GL-MRV-S3-010).

    Manages the lifecycle of provenance chains: creation, stage recording,
    chain hashing, Merkle root computation, sealing, validation, and export.

    All operations are protected by threading.RLock() to ensure thread-safe
    access in multi-threaded calculation pipelines.

    Attributes:
        agent_id: Agent identifier (GL-MRV-S3-010)
        agent_version: Agent semantic version (1.0.0)

    Example:
        >>> builder = ProvenanceChainBuilder()
        >>> chain_id = builder.start_chain()
        >>> builder.record_stage(chain_id, ProvenanceStage.VALIDATE, raw, validated)
        >>> builder.record_stage(chain_id, ProvenanceStage.CALCULATE, calc_in, calc_out)
        >>> chain = builder.get_chain(chain_id)
        >>> sealed_hash = builder.seal_chain(chain)
        >>> assert builder.validate_chain(chain)
    """

    _instance: Optional["ProvenanceChainBuilder"] = None
    _lock: threading.RLock = threading.RLock()

    def __new__(cls) -> "ProvenanceChainBuilder":
        """Thread-safe singleton instantiation using double-checked locking."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the provenance chain builder (only once due to singleton)."""
        if hasattr(self, "_initialized"):
            return

        self._initialized: bool = True
        self.agent_id: str = AGENT_ID
        self.agent_version: str = AGENT_VERSION
        self._chains: Dict[str, ProvenanceChain] = {}
        self._internal_lock: threading.RLock = threading.RLock()

        logger.info(
            "ProvenanceChainBuilder initialized for %s v%s",
            self.agent_id, self.agent_version,
        )

    # ======================================================================
    # Chain lifecycle
    # ======================================================================

    def start_chain(
        self,
        chain_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> str:
        """
        Start a new provenance chain.

        Args:
            chain_id: Optional chain ID (UUID generated if not provided)
            tenant_id: Optional tenant identifier for multi-tenant isolation

        Returns:
            Chain ID (UUID string)

        Raises:
            ValueError: If chain_id already exists
        """
        with self._internal_lock:
            if chain_id is None:
                chain_id = str(uuid.uuid4())

            if chain_id in self._chains:
                raise ValueError(f"Chain {chain_id} already exists")

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
        stage: Union[str, ProvenanceStage],
        input_data: Any,
        output_data: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StageRecord:
        """
        Record a pipeline stage in the provenance chain.

        Computes SHA-256 hashes of the input and output data and creates
        a StageRecord linked to the previous stage in the chain.

        Args:
            chain_id: Chain identifier
            stage: Pipeline stage (ProvenanceStage enum or string)
            input_data: Input data for this stage (any hashable data)
            output_data: Output data from this stage (any hashable data)
            metadata: Optional metadata dictionary

        Returns:
            Created StageRecord

        Raises:
            ValueError: If chain not found or already sealed
        """
        with self._internal_lock:
            chain = self._chains.get(chain_id)
            if chain is None:
                raise ValueError(f"Chain {chain_id} not found")
            if chain.is_sealed:
                raise ValueError(f"Chain {chain_id} is already sealed")

            stage_enum = stage if isinstance(stage, ProvenanceStage) else ProvenanceStage(stage)

            input_hash = _compute_hash(input_data)
            output_hash = _compute_hash(output_data)
            timestamp = datetime.now(timezone.utc).isoformat()

            record = StageRecord(
                stage=stage_enum,
                input_hash=input_hash,
                output_hash=output_hash,
                timestamp=timestamp,
                metadata=metadata or {},
            )

            chain.stages.append(record)

            logger.debug(
                "Recorded stage %s in chain %s (input=%s..., output=%s...)",
                stage_enum.value, chain_id, input_hash[:12], output_hash[:12],
            )
            return record

    def get_chain(self, chain_id: str) -> ProvenanceChain:
        """
        Get a provenance chain by ID.

        Args:
            chain_id: Chain identifier

        Returns:
            ProvenanceChain object

        Raises:
            ValueError: If chain not found
        """
        with self._internal_lock:
            chain = self._chains.get(chain_id)
            if chain is None:
                raise ValueError(f"Chain {chain_id} not found")
            return chain

    def get_all_chain_ids(self) -> List[str]:
        """
        Get list of all chain IDs managed by this builder.

        Returns:
            List of chain ID strings
        """
        with self._internal_lock:
            return list(self._chains.keys())

    def delete_chain(self, chain_id: str) -> bool:
        """
        Delete a provenance chain.

        Args:
            chain_id: Chain identifier

        Returns:
            True if deleted, False if not found
        """
        with self._internal_lock:
            if chain_id in self._chains:
                del self._chains[chain_id]
                logger.debug("Deleted provenance chain: %s", chain_id)
                return True
            return False

    # ======================================================================
    # Hash computation methods
    # ======================================================================

    def hash_input(self, data: dict) -> str:
        """
        Hash arbitrary input data using SHA-256 of canonical JSON.

        Args:
            data: Input data dictionary

        Returns:
            64-character lowercase hex SHA-256 hash
        """
        return _compute_hash(data)

    def hash_emission_factor(
        self,
        ef_id: str,
        ef_value: Union[Decimal, float],
        source: str,
        region: str,
    ) -> str:
        """
        Hash an emission factor for provenance tracking.

        Args:
            ef_id: Emission factor identifier
            ef_value: Emission factor value (kgCO2e per unit)
            source: Data source (defra/epa/ecoinvent/iea/beis/customer)
            region: Geographic region code

        Returns:
            64-character lowercase hex SHA-256 hash
        """
        data = {
            "ef_id": ef_id,
            "ef_value": str(ef_value),
            "source": source,
            "region": region,
        }
        return _compute_hash(data)

    def hash_product(
        self,
        product_id: str,
        category: str,
        quantity: Union[Decimal, float],
        unit: str,
    ) -> str:
        """
        Hash an intermediate product for provenance tracking.

        Args:
            product_id: Unique product identifier
            category: Intermediate product category (metals_ferrous/etc.)
            quantity: Product quantity
            unit: Unit of measure (tonne/kg/m3/etc.)

        Returns:
            64-character lowercase hex SHA-256 hash
        """
        data = {
            "product_id": product_id,
            "category": category,
            "quantity": str(quantity),
            "unit": unit,
        }
        return _compute_hash(data)

    def hash_calculation(
        self,
        method: str,
        inputs_hash: str,
        emissions_kg: Union[Decimal, float],
        ef_hash: str,
    ) -> str:
        """
        Hash a calculation result for provenance tracking.

        Args:
            method: Calculation method used (site_specific_direct/energy/fuel/
                    average_data/spend_based)
            inputs_hash: SHA-256 hash of all calculation inputs
            emissions_kg: Total emissions in kgCO2e
            ef_hash: SHA-256 hash of the emission factor used

        Returns:
            64-character lowercase hex SHA-256 hash
        """
        data = {
            "method": method,
            "inputs_hash": inputs_hash,
            "emissions_kg": str(emissions_kg),
            "ef_hash": ef_hash,
        }
        return _compute_hash(data)

    def hash_product_breakdown(
        self,
        product_id: str,
        emissions_kg: Union[Decimal, float],
        ef_used: str,
        method: str,
    ) -> str:
        """
        Hash a per-product emissions breakdown for provenance tracking.

        Args:
            product_id: Unique product identifier
            emissions_kg: Emissions attributed to this product in kgCO2e
            ef_used: Emission factor identifier or hash used
            method: Calculation method applied

        Returns:
            64-character lowercase hex SHA-256 hash
        """
        data = {
            "product_id": product_id,
            "emissions_kg": str(emissions_kg),
            "ef_used": ef_used,
            "method": method,
        }
        return _compute_hash(data)

    def hash_allocation(
        self,
        method: str,
        weights: Dict[str, Union[Decimal, float]],
        allocated_values: Dict[str, Union[Decimal, float]],
    ) -> str:
        """
        Hash an allocation operation for provenance tracking.

        Args:
            method: Allocation method (mass/revenue/units/equal)
            weights: Allocation weights by product ID
            allocated_values: Allocated emissions by product ID

        Returns:
            64-character lowercase hex SHA-256 hash
        """
        data = {
            "method": method,
            "weights": {k: str(v) for k, v in sorted(weights.items())},
            "allocated_values": {k: str(v) for k, v in sorted(allocated_values.items())},
        }
        return _compute_hash(data)

    def hash_aggregation(
        self,
        period: str,
        total_tco2e: Union[Decimal, float],
        breakdowns_hash: str,
    ) -> str:
        """
        Hash an aggregation result for provenance tracking.

        Args:
            period: Reporting period (e.g., "2025-Q4", "2025")
            total_tco2e: Total emissions in tCO2e
            breakdowns_hash: SHA-256 hash of the detailed breakdowns

        Returns:
            64-character lowercase hex SHA-256 hash
        """
        data = {
            "period": period,
            "total_tco2e": str(total_tco2e),
            "breakdowns_hash": breakdowns_hash,
        }
        return _compute_hash(data)

    def hash_compliance(
        self,
        framework: str,
        status: str,
        rules_passed: int,
        rules_failed: int,
    ) -> str:
        """
        Hash a compliance check result for provenance tracking.

        Args:
            framework: Compliance framework (ghg_protocol/iso_14064/csrd/
                       cdp/sbti/sb_253/gri)
            status: Compliance status (compliant/non_compliant/etc.)
            rules_passed: Number of rules that passed
            rules_failed: Number of rules that failed

        Returns:
            64-character lowercase hex SHA-256 hash
        """
        data = {
            "framework": framework,
            "status": status,
            "rules_passed": rules_passed,
            "rules_failed": rules_failed,
        }
        return _compute_hash(data)

    def hash_dqi(
        self,
        reliability: Union[Decimal, float],
        completeness: Union[Decimal, float],
        temporal: Union[Decimal, float],
        geographical: Union[Decimal, float],
        technological: Union[Decimal, float],
    ) -> str:
        """
        Hash a Data Quality Indicator (DQI) assessment for provenance tracking.

        Args:
            reliability: Reliability score (1.0-5.0)
            completeness: Completeness score (1.0-5.0)
            temporal: Temporal representativeness score (1.0-5.0)
            geographical: Geographical representativeness score (1.0-5.0)
            technological: Technological representativeness score (1.0-5.0)

        Returns:
            64-character lowercase hex SHA-256 hash
        """
        data = {
            "reliability": str(reliability),
            "completeness": str(completeness),
            "temporal": str(temporal),
            "geographical": str(geographical),
            "technological": str(technological),
        }
        return _compute_hash(data)

    def hash_uncertainty(
        self,
        method: str,
        mean: Union[Decimal, float],
        ci_lower: Union[Decimal, float],
        ci_upper: Union[Decimal, float],
    ) -> str:
        """
        Hash an uncertainty quantification result for provenance tracking.

        Args:
            method: Uncertainty method (analytical/monte_carlo/bootstrap/ipcc_default)
            mean: Mean emissions estimate (kgCO2e)
            ci_lower: Lower confidence interval bound (kgCO2e)
            ci_upper: Upper confidence interval bound (kgCO2e)

        Returns:
            64-character lowercase hex SHA-256 hash
        """
        data = {
            "method": method,
            "mean": str(mean),
            "ci_lower": str(ci_lower),
            "ci_upper": str(ci_upper),
        }
        return _compute_hash(data)

    def hash_stage(
        self,
        stage: Union[str, ProvenanceStage],
        input_hash: str,
        output_hash: str,
        timestamp: str,
    ) -> str:
        """
        Hash a single stage record for provenance tracking.

        Args:
            stage: Pipeline stage name or enum
            input_hash: SHA-256 hash of stage input
            output_hash: SHA-256 hash of stage output
            timestamp: ISO 8601 UTC timestamp

        Returns:
            64-character lowercase hex SHA-256 hash
        """
        stage_str = stage.value if isinstance(stage, ProvenanceStage) else stage
        data = {
            "stage": stage_str,
            "input_hash": input_hash,
            "output_hash": output_hash,
            "timestamp": timestamp,
        }
        return _compute_hash(data)

    def compute_chain_hash(self, stage_hashes: List[str]) -> str:
        """
        Compute sequential chain hash from a list of stage hashes.

        Each hash is chained to the previous by computing
        SHA-256(accumulated_hash | current_hash). The result is the
        final accumulated hash after processing all stages.

        Args:
            stage_hashes: Ordered list of SHA-256 stage hashes

        Returns:
            64-character lowercase hex SHA-256 chain hash
        """
        if not stage_hashes:
            return hashlib.sha256(b"").hexdigest()

        accumulated = stage_hashes[0]
        for i in range(1, len(stage_hashes)):
            combined = f"{accumulated}|{stage_hashes[i]}"
            accumulated = hashlib.sha256(combined.encode(ENCODING)).hexdigest()

        return accumulated

    def compute_merkle_root(self, hashes: List[str]) -> str:
        """
        Compute Merkle tree root hash from a list of hashes.

        Builds a binary Merkle tree with SHA-256 pair hashing. If the
        number of leaves at any level is odd, the last leaf is duplicated.
        Hashes are sorted before tree construction for determinism.

        Args:
            hashes: List of SHA-256 hashes (leaves of the Merkle tree)

        Returns:
            64-character lowercase hex SHA-256 Merkle root hash
        """
        if not hashes:
            return hashlib.sha256(b"").hexdigest()

        if len(hashes) == 1:
            return hashes[0]

        sorted_hashes = sorted(hashes)
        _, root = _build_merkle_tree(sorted_hashes)
        return root

    # ======================================================================
    # Stage record and chain building
    # ======================================================================

    def build_stage_record(
        self,
        stage: Union[str, ProvenanceStage],
        input_data: Any,
        output_data: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StageRecord:
        """
        Build a StageRecord from raw input/output data.

        Computes SHA-256 hashes of the input and output and creates a
        frozen StageRecord. This method does not add the record to any
        chain -- use ``record_stage()`` for that.

        Args:
            stage: Pipeline stage (ProvenanceStage enum or string)
            input_data: Raw input data for this stage
            output_data: Raw output data from this stage
            metadata: Optional metadata dictionary

        Returns:
            Frozen StageRecord with computed hashes
        """
        stage_enum = stage if isinstance(stage, ProvenanceStage) else ProvenanceStage(stage)
        input_hash = _compute_hash(input_data)
        output_hash = _compute_hash(output_data)
        timestamp = datetime.now(timezone.utc).isoformat()

        return StageRecord(
            stage=stage_enum,
            input_hash=input_hash,
            output_hash=output_hash,
            timestamp=timestamp,
            metadata=metadata or {},
        )

    def build_chain(self, stages: List[StageRecord]) -> ProvenanceChain:
        """
        Build a ProvenanceChain from a list of StageRecords.

        Computes the sequential chain hash and Merkle root from the
        stage records and returns a fully populated ProvenanceChain.
        The chain is NOT sealed -- call ``seal_chain()`` to seal it.

        Args:
            stages: Ordered list of StageRecord objects

        Returns:
            ProvenanceChain with computed chain_hash and merkle_root
        """
        chain_id = str(uuid.uuid4())

        # Compute per-stage hashes
        stage_hashes = [
            self.hash_stage(
                s.stage,
                s.input_hash,
                s.output_hash,
                s.timestamp,
            )
            for s in stages
        ]

        chain_hash = self.compute_chain_hash(stage_hashes)
        merkle_root = self.compute_merkle_root(stage_hashes)

        chain = ProvenanceChain(
            chain_id=chain_id,
            stages=list(stages),
            chain_hash=chain_hash,
            merkle_root=merkle_root,
            sealed_at="",
            agent_id=self.agent_id,
        )

        with self._internal_lock:
            self._chains[chain_id] = chain

        logger.debug(
            "Built provenance chain %s with %d stages (hash=%s...)",
            chain_id, len(stages), chain_hash[:12],
        )
        return chain

    # ======================================================================
    # Chain sealing and validation
    # ======================================================================

    def seal_chain(self, chain: ProvenanceChain) -> str:
        """
        Seal a provenance chain, making it immutable.

        Recomputes the chain hash and Merkle root from the current stages,
        sets the sealed_at timestamp, and returns the final immutable hash.
        Once sealed, no further stages can be added.

        Args:
            chain: ProvenanceChain to seal

        Returns:
            64-character lowercase hex SHA-256 final chain hash

        Raises:
            ValueError: If chain is already sealed or has no stages
        """
        with self._internal_lock:
            if chain.is_sealed:
                raise ValueError(f"Chain {chain.chain_id} is already sealed")

            if not chain.stages:
                raise ValueError(f"Chain {chain.chain_id} has no stages to seal")

            # Recompute hashes from current stages
            stage_hashes = [
                self.hash_stage(
                    s.stage,
                    s.input_hash,
                    s.output_hash,
                    s.timestamp,
                )
                for s in chain.stages
            ]

            chain.chain_hash = self.compute_chain_hash(stage_hashes)
            chain.merkle_root = self.compute_merkle_root(stage_hashes)
            chain.sealed_at = datetime.now(timezone.utc).isoformat()

            logger.info(
                "Sealed provenance chain %s with %d stages (hash=%s)",
                chain.chain_id, len(chain.stages), chain.chain_hash,
            )
            return chain.chain_hash

    def validate_chain(self, chain: ProvenanceChain) -> bool:
        """
        Validate integrity of a provenance chain.

        Performs the following checks:
        1. Chain has at least one stage
        2. Chain hash matches recomputed value from stages
        3. Merkle root matches recomputed value from stages
        4. All stage hashes are valid 64-character hex strings
        5. Stage sequence follows valid ordering

        Args:
            chain: ProvenanceChain to validate

        Returns:
            True if chain integrity is verified, False otherwise
        """
        with self._internal_lock:
            # Check 1: Non-empty stages
            if not chain.stages:
                logger.warning(
                    "Chain %s validation failed: no stages", chain.chain_id,
                )
                return False

            # Check 2: All hashes are valid hex strings
            for i, stage_record in enumerate(chain.stages):
                if not self._is_valid_hash(stage_record.input_hash):
                    logger.warning(
                        "Chain %s validation failed: invalid input_hash at stage %d",
                        chain.chain_id, i,
                    )
                    return False
                if not self._is_valid_hash(stage_record.output_hash):
                    logger.warning(
                        "Chain %s validation failed: invalid output_hash at stage %d",
                        chain.chain_id, i,
                    )
                    return False

            # Check 3: Recompute chain hash and compare
            stage_hashes = [
                self.hash_stage(
                    s.stage,
                    s.input_hash,
                    s.output_hash,
                    s.timestamp,
                )
                for s in chain.stages
            ]

            expected_chain_hash = self.compute_chain_hash(stage_hashes)
            if chain.chain_hash != expected_chain_hash:
                logger.warning(
                    "Chain %s validation failed: chain_hash mismatch "
                    "(expected=%s, actual=%s)",
                    chain.chain_id, expected_chain_hash, chain.chain_hash,
                )
                return False

            # Check 4: Recompute Merkle root and compare
            expected_merkle = self.compute_merkle_root(stage_hashes)
            if chain.merkle_root != expected_merkle:
                logger.warning(
                    "Chain %s validation failed: merkle_root mismatch "
                    "(expected=%s, actual=%s)",
                    chain.chain_id, expected_merkle, chain.merkle_root,
                )
                return False

            # Check 5: Validate stage sequence
            if not self.validate_stage_sequence(chain.stages):
                logger.warning(
                    "Chain %s validation failed: invalid stage sequence",
                    chain.chain_id,
                )
                return False

            logger.debug(
                "Chain %s validation passed (%d stages)",
                chain.chain_id, len(chain.stages),
            )
            return True

    def validate_stage_sequence(self, stages: List[StageRecord]) -> bool:
        """
        Validate that pipeline stages are in a valid order.

        Stages must appear in non-decreasing order according to the
        canonical pipeline stage ordering (VALIDATE=0, CLASSIFY=1, ...,
        SEAL=9). Gaps are allowed (not every stage is required), but
        stages must not go backwards.

        Args:
            stages: Ordered list of StageRecord objects

        Returns:
            True if stage ordering is valid, False otherwise
        """
        if not stages:
            return True

        last_order = -1
        for record in stages:
            stage_str = (
                record.stage.value
                if isinstance(record.stage, ProvenanceStage)
                else str(record.stage)
            )
            current_order = _STAGE_ORDER.get(stage_str, -1)

            if current_order == -1:
                logger.warning(
                    "Unknown stage '%s' in sequence validation", stage_str,
                )
                return False

            if current_order < last_order:
                logger.warning(
                    "Stage '%s' (order=%d) appears after stage with order=%d",
                    stage_str, current_order, last_order,
                )
                return False

            last_order = current_order

        return True

    def export_chain(self, chain: ProvenanceChain) -> dict:
        """
        Export a provenance chain as a JSON-serializable dictionary.

        Includes all stage records, hashes, timestamps, and metadata
        suitable for persistence in a database or audit log.

        Args:
            chain: ProvenanceChain to export

        Returns:
            JSON-serializable dictionary representation of the chain
        """
        return chain.to_dict()

    # ======================================================================
    # Batch provenance support
    # ======================================================================

    def start_batch(
        self,
        item_count: int = 0,
        batch_id: Optional[str] = None,
    ) -> str:
        """
        Start a new batch provenance session.

        Args:
            item_count: Expected number of items in this batch
            batch_id: Optional batch ID (UUID generated if not provided)

        Returns:
            Batch ID string

        Raises:
            ValueError: If batch_id already exists
        """
        with self._internal_lock:
            if batch_id is None:
                batch_id = str(uuid.uuid4())

            if not hasattr(self, "_batches"):
                self._batches: Dict[str, BatchProvenance] = {}

            if batch_id in self._batches:
                raise ValueError(f"Batch {batch_id} already exists")

            batch = BatchProvenance(batch_id=batch_id, item_count=item_count)
            self._batches[batch_id] = batch

            logger.debug("Started batch: %s (expected %d items)", batch_id, item_count)
            return batch_id

    def add_chain_to_batch(self, batch_id: str, chain_id: str) -> None:
        """
        Add a chain to a batch.

        Args:
            batch_id: Batch identifier
            chain_id: Chain identifier to add

        Raises:
            ValueError: If batch not found or already sealed
        """
        with self._internal_lock:
            if not hasattr(self, "_batches"):
                self._batches = {}

            batch = self._batches.get(batch_id)
            if batch is None:
                raise ValueError(f"Batch {batch_id} not found")
            if batch.batch_sealed_at is not None:
                raise ValueError(f"Batch {batch_id} already sealed")

            batch.individual_chain_ids.append(chain_id)

    def seal_batch(self, batch_id: str) -> str:
        """
        Seal the batch with a Merkle tree root hash.

        Collects all chain hashes from the batch, builds a Merkle tree,
        and seals the batch with the root hash.

        Args:
            batch_id: Batch identifier

        Returns:
            Batch Merkle root hash

        Raises:
            ValueError: If batch not found or already sealed
        """
        with self._internal_lock:
            if not hasattr(self, "_batches"):
                self._batches = {}

            batch = self._batches.get(batch_id)
            if batch is None:
                raise ValueError(f"Batch {batch_id} not found")
            if batch.batch_sealed_at is not None:
                raise ValueError(f"Batch {batch_id} already sealed")

            # Collect chain hashes
            leaf_hashes = []
            for cid in batch.individual_chain_ids:
                chain = self._chains.get(cid)
                if chain is not None and chain.chain_hash:
                    leaf_hashes.append(chain.chain_hash)

            if not leaf_hashes:
                leaf_hashes = [hashlib.sha256(b"empty").hexdigest()]

            tree_levels, root_hash = _build_merkle_tree(sorted(leaf_hashes))

            batch.leaf_hashes = sorted(leaf_hashes)
            batch.tree_levels = tree_levels
            batch.root_hash = root_hash
            batch.batch_hash = root_hash
            batch.batch_sealed_at = datetime.now(timezone.utc).isoformat()

            logger.info(
                "Sealed batch %s with %d chains (root=%s)",
                batch_id, len(leaf_hashes), root_hash,
            )
            return root_hash

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
        with self._internal_lock:
            if not hasattr(self, "_batches"):
                self._batches = {}

            batch = self._batches.get(batch_id)
            if batch is None:
                raise ValueError(f"Batch {batch_id} not found")

            return batch.to_dict()

    # ======================================================================
    # Reset / cleanup
    # ======================================================================

    def reset(self) -> None:
        """
        Clear all chains and batches. Useful for testing.

        WARNING: Destroys all provenance data. Only use in test teardown.
        """
        with self._internal_lock:
            self._chains.clear()
            if hasattr(self, "_batches"):
                self._batches.clear()
            logger.info("ProvenanceChainBuilder reset: all chains cleared")

    @classmethod
    def reset_singleton(cls) -> None:
        """
        Reset the singleton instance for testing purposes.

        Destroys the current singleton so the next call to the constructor
        or ``get_provenance_builder()`` creates a fresh instance.

        WARNING: Not safe for concurrent use. Only call in test teardown.
        """
        with cls._lock:
            if cls._instance is not None:
                if hasattr(cls._instance, "_initialized"):
                    del cls._instance._initialized
                cls._instance = None

                global _builder_instance
                _builder_instance = None

                logger.info("ProvenanceChainBuilder singleton reset")

    # ======================================================================
    # Internal helpers
    # ======================================================================

    @staticmethod
    def _is_valid_hash(h: str) -> bool:
        """
        Validate that a string is a valid 64-character lowercase hex SHA-256 hash.

        Args:
            h: String to validate

        Returns:
            True if valid SHA-256 hex hash
        """
        if not isinstance(h, str):
            return False
        if len(h) != 64:
            return False
        try:
            int(h, 16)
            return True
        except ValueError:
            return False


# ============================================================================
# STANDALONE HASH FUNCTIONS
# ============================================================================


def hash_product_input(product_data: Dict[str, Any]) -> str:
    """
    Hash intermediate product input data.

    Args:
        product_data: Product data dictionary (product_id, category, quantity, etc.)

    Returns:
        64-character lowercase hex SHA-256 hash
    """
    return _compute_hash(product_data)


def hash_processing_input(processing_data: Dict[str, Any]) -> str:
    """
    Hash processing operation input data.

    Args:
        processing_data: Processing data (type, energy, fuel, etc.)

    Returns:
        64-character lowercase hex SHA-256 hash
    """
    return _compute_hash(processing_data)


def hash_site_specific_result(
    method: str,
    energy_kwh: Union[Decimal, float],
    grid_factor: Union[Decimal, float],
    co2e_kg: Union[Decimal, float],
) -> str:
    """
    Hash site-specific calculation result.

    Args:
        method: Site-specific method (direct/energy/fuel)
        energy_kwh: Energy consumption in kWh
        grid_factor: Grid or fuel emission factor
        co2e_kg: Total emissions kgCO2e

    Returns:
        64-character lowercase hex SHA-256 hash
    """
    data = {
        "method": method,
        "energy_kwh": str(energy_kwh),
        "grid_factor": str(grid_factor),
        "co2e_kg": str(co2e_kg),
        "formula": "energy_kwh * grid_factor",
    }
    return _compute_hash(data)


def hash_average_data_result(
    category: str,
    processing_type: str,
    quantity_tonnes: Union[Decimal, float],
    ef_per_tonne: Union[Decimal, float],
    co2e_kg: Union[Decimal, float],
) -> str:
    """
    Hash average-data calculation result.

    Args:
        category: Intermediate product category
        processing_type: Processing type applied
        quantity_tonnes: Quantity in tonnes
        ef_per_tonne: Emission factor kgCO2e per tonne processed
        co2e_kg: Total emissions kgCO2e

    Returns:
        64-character lowercase hex SHA-256 hash
    """
    data = {
        "category": category,
        "processing_type": processing_type,
        "quantity_tonnes": str(quantity_tonnes),
        "ef_per_tonne": str(ef_per_tonne),
        "co2e_kg": str(co2e_kg),
        "formula": "quantity_tonnes * ef_per_tonne",
    }
    return _compute_hash(data)


def hash_spend_based_result(
    spend_usd: Union[Decimal, float],
    naics_code: str,
    ef: Union[Decimal, float],
    co2e_kg: Union[Decimal, float],
    margin_removed: bool,
    cpi_deflated: bool,
) -> str:
    """
    Hash spend-based calculation result.

    Args:
        spend_usd: Spend in USD (after conversion)
        naics_code: NAICS sector code
        ef: EEIO emission factor (kgCO2e per USD)
        co2e_kg: Total emissions kgCO2e
        margin_removed: Whether profit margin was removed
        cpi_deflated: Whether CPI deflation was applied

    Returns:
        64-character lowercase hex SHA-256 hash
    """
    data = {
        "spend_usd": str(spend_usd),
        "naics_code": naics_code,
        "ef": str(ef),
        "co2e_kg": str(co2e_kg),
        "margin_removed": margin_removed,
        "cpi_deflated": cpi_deflated,
        "formula": "spend_usd * ef",
    }
    return _compute_hash(data)


def hash_processing_chain_result(
    chain_type: str,
    steps: List[Dict[str, Any]],
    combined_ef: Union[Decimal, float],
    total_co2e: Union[Decimal, float],
) -> str:
    """
    Hash multi-step processing chain result.

    Args:
        chain_type: Processing chain type (metals_automotive/etc.)
        steps: List of step dictionaries
        combined_ef: Combined emission factor across all steps
        total_co2e: Total emissions across all steps kgCO2e

    Returns:
        64-character lowercase hex SHA-256 hash
    """
    data = {
        "chain_type": chain_type,
        "steps": steps,
        "combined_ef": str(combined_ef),
        "total_co2e": str(total_co2e),
    }
    return _compute_hash(data)


def hash_grid_ef(
    region: str,
    ef_value: Union[Decimal, float],
    source: str,
    year: int,
) -> str:
    """
    Hash grid electricity emission factor.

    Args:
        region: Grid region code (US/GB/DE/etc.)
        ef_value: Emission factor value (kgCO2e/kWh)
        source: Data source (iea/egrid/beis/etc.)
        year: Factor year

    Returns:
        64-character lowercase hex SHA-256 hash
    """
    data = {
        "region": region,
        "ef_value": str(ef_value),
        "source": source,
        "year": year,
    }
    return _compute_hash(data)


def hash_fuel_ef(
    fuel_type: str,
    ef_value: Union[Decimal, float],
    source: str,
) -> str:
    """
    Hash fuel combustion emission factor.

    Args:
        fuel_type: Fuel type (natural_gas/diesel/hfo/lpg/coal/biomass)
        ef_value: Emission factor value (kgCO2e/kWh thermal)
        source: Data source

    Returns:
        64-character lowercase hex SHA-256 hash
    """
    data = {
        "fuel_type": fuel_type,
        "ef_value": str(ef_value),
        "source": source,
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
        method: Allocation method (mass/revenue/units/equal)
        share: Product share (0.0-1.0)
        total: Total emissions before allocation
        allocated: Allocated emissions for this product

    Returns:
        64-character lowercase hex SHA-256 hash
    """
    data = {
        "method": method,
        "share": str(share),
        "total": str(total),
        "allocated": str(allocated),
        "formula": "total * share",
    }
    return _compute_hash(data)


def hash_aggregation_result(
    by_category: Dict[str, Any],
    by_method: Dict[str, Any],
    by_processing_type: Dict[str, Any],
    total: Union[Decimal, float],
) -> str:
    """
    Hash aggregation result.

    Args:
        by_category: Emissions by product category
        by_method: Emissions by calculation method
        by_processing_type: Emissions by processing type
        total: Total emissions

    Returns:
        64-character lowercase hex SHA-256 hash
    """
    data = {
        "by_category": by_category,
        "by_method": by_method,
        "by_processing_type": by_processing_type,
        "total": str(total),
    }
    return _compute_hash(data)


def hash_compliance_result(
    framework: str,
    status: str,
    score: Union[Decimal, float],
) -> str:
    """
    Hash compliance check result.

    Args:
        framework: Regulatory framework
        status: Compliance status
        score: Compliance score

    Returns:
        64-character lowercase hex SHA-256 hash
    """
    data = {
        "framework": framework,
        "status": status,
        "score": str(score),
    }
    return _compute_hash(data)


def hash_dc_rule_result(
    rule_id: str,
    triggered: bool,
    overlap_category: str,
    amount_excluded: Union[Decimal, float],
) -> str:
    """
    Hash double-counting prevention rule result.

    Args:
        rule_id: Double-counting rule ID (DC-PSP-001 through DC-PSP-008)
        triggered: Whether the rule was triggered
        overlap_category: The overlapping Scope/Category
        amount_excluded: Amount excluded in kgCO2e

    Returns:
        64-character lowercase hex SHA-256 hash
    """
    data = {
        "rule_id": rule_id,
        "triggered": triggered,
        "overlap_category": overlap_category,
        "amount_excluded": str(amount_excluded),
    }
    return _compute_hash(data)


def hash_dqi_result(
    dqi_scores: Dict[str, Union[Decimal, float]],
    composite: Union[Decimal, float],
) -> str:
    """
    Hash data quality assessment result.

    Args:
        dqi_scores: DQI scores by dimension (reliability, completeness, etc.)
        composite: Composite DQI score

    Returns:
        64-character lowercase hex SHA-256 hash
    """
    data = {
        "dqi_scores": {k: str(v) for k, v in sorted(dqi_scores.items())},
        "composite": str(composite),
    }
    return _compute_hash(data)


def hash_uncertainty_result(
    method: str,
    confidence_level: Union[Decimal, float],
    lower_bound: Union[Decimal, float],
    upper_bound: Union[Decimal, float],
) -> str:
    """
    Hash uncertainty quantification result.

    Args:
        method: Uncertainty method (analytical/monte_carlo/bootstrap/ipcc_default)
        confidence_level: Confidence level (e.g., 0.95)
        lower_bound: Lower confidence bound (kgCO2e)
        upper_bound: Upper confidence bound (kgCO2e)

    Returns:
        64-character lowercase hex SHA-256 hash
    """
    data = {
        "method": method,
        "confidence_level": str(confidence_level),
        "lower_bound": str(lower_bound),
        "upper_bound": str(upper_bound),
    }
    return _compute_hash(data)


def hash_energy_consumption(
    energy_type: str,
    quantity_kwh: Union[Decimal, float],
    source: str,
    period: str,
) -> str:
    """
    Hash energy consumption data for site-specific calculations.

    Args:
        energy_type: Energy type (electricity/natural_gas/diesel/etc.)
        quantity_kwh: Energy quantity in kWh
        source: Data source (meter_reading/invoice/estimate)
        period: Reporting period

    Returns:
        64-character lowercase hex SHA-256 hash
    """
    data = {
        "energy_type": energy_type,
        "quantity_kwh": str(quantity_kwh),
        "source": source,
        "period": period,
    }
    return _compute_hash(data)


def hash_fuel_consumption(
    fuel_type: str,
    quantity: Union[Decimal, float],
    unit: str,
    period: str,
) -> str:
    """
    Hash fuel consumption data for site-specific calculations.

    Args:
        fuel_type: Fuel type (natural_gas/diesel/hfo/lpg/coal/biomass)
        quantity: Fuel quantity
        unit: Unit of measure (litres/kg/m3/kWh)
        period: Reporting period

    Returns:
        64-character lowercase hex SHA-256 hash
    """
    data = {
        "fuel_type": fuel_type,
        "quantity": str(quantity),
        "unit": unit,
        "period": period,
    }
    return _compute_hash(data)


def hash_currency_conversion(
    original_amount: Union[Decimal, float],
    original_currency: str,
    converted_usd: Union[Decimal, float],
    exchange_rate: Union[Decimal, float],
    cpi_deflator: Union[Decimal, float],
) -> str:
    """
    Hash currency conversion for spend-based calculations.

    Args:
        original_amount: Original spend amount
        original_currency: Original currency code
        converted_usd: Converted amount in USD
        exchange_rate: Exchange rate used
        cpi_deflator: CPI deflation factor applied

    Returns:
        64-character lowercase hex SHA-256 hash
    """
    data = {
        "original_amount": str(original_amount),
        "original_currency": original_currency,
        "converted_usd": str(converted_usd),
        "exchange_rate": str(exchange_rate),
        "cpi_deflator": str(cpi_deflator),
    }
    return _compute_hash(data)


def hash_batch_input(
    batch_id: str,
    item_count: int,
    product_hashes: List[str],
) -> str:
    """
    Hash batch input data.

    Args:
        batch_id: Batch identifier
        item_count: Number of items in batch
        product_hashes: List of individual product input hashes

    Returns:
        64-character lowercase hex SHA-256 hash
    """
    data = {
        "batch_id": batch_id,
        "item_count": item_count,
        "product_hashes": sorted(product_hashes),
    }
    return _compute_hash(data)


def hash_batch_result(
    batch_id: str,
    results: List[Dict[str, Any]],
) -> str:
    """
    Hash batch calculation result.

    Args:
        batch_id: Batch identifier
        results: List of individual calculation result dictionaries

    Returns:
        64-character lowercase hex SHA-256 hash
    """
    data = {
        "batch_id": batch_id,
        "results": results,
    }
    return _compute_hash(data)


def hash_config(config: Dict[str, Any]) -> str:
    """
    Hash configuration data.

    Args:
        config: Configuration dictionary

    Returns:
        64-character lowercase hex SHA-256 hash
    """
    return _compute_hash(config)


def hash_metadata(metadata: Dict[str, Any]) -> str:
    """
    Hash metadata.

    Args:
        metadata: Metadata dictionary

    Returns:
        64-character lowercase hex SHA-256 hash
    """
    return _compute_hash(metadata)


def hash_arbitrary(data: Any) -> str:
    """
    Hash arbitrary data using canonical JSON serialization.

    Args:
        data: Any data to hash

    Returns:
        64-character lowercase hex SHA-256 hash
    """
    return _compute_hash(data)


# ============================================================================
# CONVENIENCE STAGE RECORDERS (10 stages)
# ============================================================================


def create_chain(tenant_id: Optional[str] = None) -> Tuple["ProvenanceChainBuilder", str]:
    """
    Create a new provenance chain using singleton builder.

    Args:
        tenant_id: Optional tenant identifier

    Returns:
        Tuple of (builder, chain_id)
    """
    builder = get_provenance_builder()
    chain_id = builder.start_chain(tenant_id=tenant_id)
    return builder, chain_id


def record_validation(
    chain_id: str,
    input_data: Any,
    validated_data: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> StageRecord:
    """
    Record VALIDATE stage in the singleton builder.

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
    Record CLASSIFY stage in the singleton builder.

    Args:
        chain_id: Chain identifier
        input_data: Input data
        classified_data: Classified data (category, processing type)
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
    Record NORMALIZE stage in the singleton builder.

    Args:
        chain_id: Chain identifier
        input_data: Input data
        normalized_data: Normalized data (units, currency)
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
    resolved_efs: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> StageRecord:
    """
    Record RESOLVE_EFS stage in the singleton builder.

    Args:
        chain_id: Chain identifier
        input_data: Input data (product category, processing type, region)
        resolved_efs: Resolved emission factors
        metadata: Optional metadata (source, year, etc.)

    Returns:
        StageRecord
    """
    builder = get_provenance_builder()
    return builder.record_stage(
        chain_id, ProvenanceStage.RESOLVE_EFS, input_data, resolved_efs, metadata
    )


def record_calculation(
    chain_id: str,
    input_data: Any,
    calculation_results: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> StageRecord:
    """
    Record CALCULATE stage in the singleton builder.

    Args:
        chain_id: Chain identifier
        input_data: Calculation input
        calculation_results: Calculation results
        metadata: Optional metadata (method, formula, etc.)

    Returns:
        StageRecord
    """
    builder = get_provenance_builder()
    return builder.record_stage(
        chain_id, ProvenanceStage.CALCULATE, input_data, calculation_results, metadata
    )


def record_allocation(
    chain_id: str,
    input_data: Any,
    allocation_results: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> StageRecord:
    """
    Record ALLOCATE stage in the singleton builder.

    Args:
        chain_id: Chain identifier
        input_data: Pre-allocation data
        allocation_results: Allocation results
        metadata: Optional metadata (method, weights, etc.)

    Returns:
        StageRecord
    """
    builder = get_provenance_builder()
    return builder.record_stage(
        chain_id, ProvenanceStage.ALLOCATE, input_data, allocation_results, metadata
    )


def record_aggregation(
    chain_id: str,
    input_data: Any,
    aggregated_results: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> StageRecord:
    """
    Record AGGREGATE stage in the singleton builder.

    Args:
        chain_id: Chain identifier
        input_data: Pre-aggregation data
        aggregated_results: Aggregated results
        metadata: Optional metadata (dimensions, totals, etc.)

    Returns:
        StageRecord
    """
    builder = get_provenance_builder()
    return builder.record_stage(
        chain_id, ProvenanceStage.AGGREGATE, input_data, aggregated_results, metadata
    )


def record_compliance(
    chain_id: str,
    input_data: Any,
    compliance_results: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> StageRecord:
    """
    Record COMPLIANCE stage in the singleton builder.

    Args:
        chain_id: Chain identifier
        input_data: Data to check
        compliance_results: Compliance check results
        metadata: Optional metadata (framework, rules, etc.)

    Returns:
        StageRecord
    """
    builder = get_provenance_builder()
    return builder.record_stage(
        chain_id, ProvenanceStage.COMPLIANCE, input_data, compliance_results, metadata
    )


def record_provenance(
    chain_id: str,
    input_data: Any,
    provenance_results: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> StageRecord:
    """
    Record PROVENANCE stage in the singleton builder.

    Args:
        chain_id: Chain identifier
        input_data: Data to hash
        provenance_results: Provenance hash results
        metadata: Optional metadata

    Returns:
        StageRecord
    """
    builder = get_provenance_builder()
    return builder.record_stage(
        chain_id, ProvenanceStage.PROVENANCE, input_data, provenance_results, metadata
    )


def seal_and_verify(chain_id: str) -> Tuple[str, bool]:
    """
    Seal chain and verify integrity using the singleton builder.

    Args:
        chain_id: Chain identifier

    Returns:
        Tuple of (final_hash, is_valid)
    """
    builder = get_provenance_builder()
    chain = builder.get_chain(chain_id)

    # Record SEAL stage
    seal_input = {"chain_id": chain_id, "stage_count": len(chain.stages)}
    seal_output = {"action": "seal", "timestamp": datetime.now(timezone.utc).isoformat()}
    builder.record_stage(chain_id, ProvenanceStage.SEAL, seal_input, seal_output)

    # Now build the final chain and seal
    chain = builder.get_chain(chain_id)
    stage_hashes = [
        builder.hash_stage(s.stage, s.input_hash, s.output_hash, s.timestamp)
        for s in chain.stages
    ]
    chain.chain_hash = builder.compute_chain_hash(stage_hashes)
    chain.merkle_root = builder.compute_merkle_root(stage_hashes)

    final_hash = builder.seal_chain(chain)
    is_valid = builder.validate_chain(chain)
    return final_hash, is_valid


def export_chain_json(chain_id: str) -> str:
    """
    Export chain as JSON string.

    Args:
        chain_id: Chain identifier

    Returns:
        JSON string representation of the chain
    """
    builder = get_provenance_builder()
    chain = builder.get_chain(chain_id)
    return chain.to_json()


# ============================================================================
# SINGLETON ACCESS
# ============================================================================

_builder_instance: Optional[ProvenanceChainBuilder] = None
_builder_lock: threading.RLock = threading.RLock()


def get_provenance_builder() -> ProvenanceChainBuilder:
    """
    Get the singleton ProvenanceChainBuilder instance.

    Thread-safe accessor. Creates the singleton on first call.

    Returns:
        ProvenanceChainBuilder singleton instance

    Example:
        >>> builder = get_provenance_builder()
        >>> chain_id = builder.start_chain()
        >>> builder.record_stage(chain_id, ProvenanceStage.VALIDATE, raw, validated)
    """
    global _builder_instance

    if _builder_instance is None:
        with _builder_lock:
            if _builder_instance is None:
                _builder_instance = ProvenanceChainBuilder()

    return _builder_instance


def reset_provenance_builder() -> None:
    """
    Reset the singleton ProvenanceChainBuilder instance for testing purposes.

    Destroys the current singleton so the next call to
    ``get_provenance_builder()`` creates a fresh instance.

    Example:
        >>> reset_provenance_builder()
        >>> builder = get_provenance_builder()  # brand new instance
    """
    ProvenanceChainBuilder.reset_singleton()


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "ProvenanceStage",
    # Models
    "StageRecord",
    "ProvenanceChain",
    "BatchProvenance",
    # Builder class
    "ProvenanceChainBuilder",
    # Singletons
    "get_provenance_builder",
    "reset_provenance_builder",
    # Standalone hash functions (25+)
    "hash_product_input",
    "hash_processing_input",
    "hash_site_specific_result",
    "hash_average_data_result",
    "hash_spend_based_result",
    "hash_processing_chain_result",
    "hash_grid_ef",
    "hash_fuel_ef",
    "hash_allocation_result",
    "hash_aggregation_result",
    "hash_compliance_result",
    "hash_dc_rule_result",
    "hash_dqi_result",
    "hash_uncertainty_result",
    "hash_energy_consumption",
    "hash_fuel_consumption",
    "hash_currency_conversion",
    "hash_batch_input",
    "hash_batch_result",
    "hash_config",
    "hash_metadata",
    "hash_arbitrary",
    # Convenience stage recorders
    "create_chain",
    "record_validation",
    "record_classification",
    "record_normalization",
    "record_ef_resolution",
    "record_calculation",
    "record_allocation",
    "record_aggregation",
    "record_compliance",
    "record_provenance",
    "seal_and_verify",
    "export_chain_json",
    # Low-level utilities
    "_compute_hash",
    "_serialize",
    "_build_merkle_tree",
    "_verify_merkle_proof",
]
