# -*- coding: utf-8 -*-
"""
Scope 3 Category Mapper Provenance Tracking - AGENT-MRV-029

SHA-256 chain-hashed provenance for zero-hallucination audit trails.
10 pipeline stages tracked with deterministic hashing.
Agent: GL-MRV-X-040

This module implements a complete provenance tracking system for
Scope 3 category classification decisions. Every step in the 10-stage
pipeline is recorded with SHA-256 hashes, creating an immutable audit
trail that proves no classification was hallucinated or modified.

Pipeline Stages:
    1. VALIDATE         - Input validation and normalization
    2. SOURCE_CLASSIFY  - Data source classification
    3. CODE_LOOKUP      - Industry code lookup (NAICS/ISIC/UNSPSC/HS)
    4. SPEND_CLASSIFY   - Spend classification (GL/procurement/keyword)
    5. BOUNDARY         - Boundary determination (upstream/downstream)
    6. DOUBLE_COUNTING  - Cross-category double-counting check
    7. SPLIT            - Multi-category splitting
    8. RECOMMEND        - Calculation approach recommendation
    9. COMPLETENESS     - Completeness screening
    10. SEAL            - Final sealing and verification

10 Hash Functions:
    1. hash_raw_input()            - Hash raw input record
    2. hash_code_lookup()          - Hash NAICS/ISIC/UNSPSC lookup
    3. hash_classification()       - Hash category + method + confidence
    4. hash_boundary()             - Hash upstream/downstream result
    5. hash_double_counting()      - Hash DC check analysis
    6. hash_split_allocation()     - Hash multi-category split ratios
    7. hash_approach()             - Hash calculation approach
    8. hash_routing()              - Hash target agent + input
    9. hash_completeness()         - Hash category coverage impact
    10. build_chain_hash()         - SHA-256(P1 || P2 || ... || P9)

Example:
    >>> engine = ProvenanceEngine()
    >>> record = engine.create_provenance_record(
    ...     PipelineStage.VALIDATE, input_data, validated_data
    ... )
    >>> chain = engine.build_full_chain([record])
    >>> assert engine.verify_chain(chain)

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-X-040
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

AGENT_ID = "GL-MRV-X-040"
AGENT_VERSION = "1.0.0"
HASH_ALGORITHM = "sha256"
ENCODING = "utf-8"


class PipelineStage(str, Enum):
    """Pipeline stages for category mapper provenance tracking."""

    VALIDATE = "VALIDATE"
    SOURCE_CLASSIFY = "SOURCE_CLASSIFY"
    CODE_LOOKUP = "CODE_LOOKUP"
    SPEND_CLASSIFY = "SPEND_CLASSIFY"
    BOUNDARY = "BOUNDARY"
    DOUBLE_COUNTING = "DOUBLE_COUNTING"
    SPLIT = "SPLIT"
    RECOMMEND = "RECOMMEND"
    COMPLETENESS = "COMPLETENESS"
    SEAL = "SEAL"


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass(frozen=True)
class ProvenanceRecord:
    """
    Single provenance record in the chain.

    Frozen (immutable) record of one stage in the classification pipeline.
    The stage_hash links the input and output data with the pipeline
    stage, creating an auditable trail.

    Attributes:
        record_id: Unique identifier for this record (UUID).
        stage: Pipeline stage name (from PipelineStage enum).
        input_hash: SHA-256 hash of input data to this stage.
        output_hash: SHA-256 hash of output data from this stage.
        stage_hash: SHA-256 hash combining stage + input_hash + output_hash.
        timestamp: ISO 8601 UTC timestamp of record creation.
        agent_id: Agent identifier (GL-MRV-X-040).
        agent_version: Agent version (1.0.0).
        metadata: Additional context (optional).
    """

    record_id: str
    stage: str
    input_hash: str
    output_hash: str
    stage_hash: str
    timestamp: str
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
    Complete provenance chain for a classification run.

    Tracks all records in pipeline order with chain-level hash.

    Attributes:
        chain_id: Unique identifier for this chain.
        records: Ordered list of provenance records.
        chain_hash: Final SHA-256 chain hash (all stage hashes combined).
        created_at: ISO 8601 UTC timestamp of chain creation.
        agent_id: Agent identifier (GL-MRV-X-040).
    """

    chain_id: str
    records: List[ProvenanceRecord] = field(default_factory=list)
    chain_hash: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    agent_id: str = AGENT_ID

    @property
    def is_valid(self) -> bool:
        """Whether the chain has been computed and has records."""
        return len(self.records) > 0 and self.chain_hash != ""

    @property
    def stage_count(self) -> int:
        """Number of stages recorded."""
        return len(self.records)

    def to_dict(self) -> Dict[str, Any]:
        """Convert chain to dictionary."""
        return {
            "chain_id": self.chain_id,
            "records": [r.to_dict() for r in self.records],
            "chain_hash": self.chain_hash,
            "created_at": self.created_at,
            "agent_id": self.agent_id,
            "is_valid": self.is_valid,
            "stage_count": self.stage_count,
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


def _compute_stage_hash(stage: str, input_hash: str, output_hash: str) -> str:
    """
    Compute stage hash combining stage name with input and output hashes.

    Stage hash = SHA-256(stage + "|" + input_hash + "|" + output_hash)

    Args:
        stage: Pipeline stage name.
        input_hash: Hash of input data.
        output_hash: Hash of output data.

    Returns:
        SHA-256 stage hash.
    """
    stage_data = f"{stage}|{input_hash}|{output_hash}"
    return hashlib.sha256(stage_data.encode(ENCODING)).hexdigest()


def _compute_chain_hash(stage_hashes: List[str]) -> str:
    """
    Compute chain hash from ordered list of stage hashes.

    Chain hash = SHA-256(stage_hash_1 || stage_hash_2 || ... || stage_hash_n)

    Args:
        stage_hashes: Ordered list of stage hashes.

    Returns:
        SHA-256 chain hash.
    """
    if not stage_hashes:
        return hashlib.sha256(b"").hexdigest()
    combined = "||".join(stage_hashes)
    return hashlib.sha256(combined.encode(ENCODING)).hexdigest()


# ============================================================================
# PROVENANCE ENGINE
# ============================================================================


class ProvenanceEngine:
    """
    Provenance tracking engine for Scope 3 Category Mapper.

    Manages creation, chaining, and verification of provenance records
    across the 10-stage classification pipeline. Thread-safe with RLock.

    Attributes:
        agent_id: Agent identifier.
        agent_version: Agent version string.

    Example:
        >>> engine = ProvenanceEngine()
        >>> r1 = engine.create_provenance_record(
        ...     PipelineStage.VALIDATE, raw_input, validated_input
        ... )
        >>> r2 = engine.create_provenance_record(
        ...     PipelineStage.CODE_LOOKUP, validated_input, lookup_result
        ... )
        >>> chain = engine.build_full_chain([r1, r2])
        >>> assert engine.verify_chain(chain)
    """

    def __init__(
        self,
        agent_id: str = AGENT_ID,
        agent_version: str = AGENT_VERSION,
    ) -> None:
        """
        Initialize the provenance engine.

        Args:
            agent_id: Agent identifier.
            agent_version: Agent version string.
        """
        self.agent_id = agent_id
        self.agent_version = agent_version
        self._lock = threading.RLock()

    def compute_stage_hash(self, stage: PipelineStage, data: Any) -> str:
        """
        Compute SHA-256 hash for a single pipeline stage.

        Convenience method that hashes the stage name combined with the
        data payload.

        Args:
            stage: Pipeline stage enum value.
            data: Data to hash (any serializable type).

        Returns:
            SHA-256 hex digest string.
        """
        stage_str = stage.value if isinstance(stage, PipelineStage) else str(stage)
        data_hash = _compute_hash(data)
        combined = f"{stage_str}|{data_hash}"
        return hashlib.sha256(combined.encode(ENCODING)).hexdigest()

    def compute_chain_hash(self, stage_hashes: List[str]) -> str:
        """
        Compute chain hash from an ordered list of stage hashes.

        Deterministically combines all stage hashes into a single
        chain-level hash.

        Args:
            stage_hashes: Ordered list of SHA-256 stage hashes.

        Returns:
            SHA-256 chain hash.
        """
        return _compute_chain_hash(stage_hashes)

    def create_provenance_record(
        self,
        stage: PipelineStage,
        input_data: Any,
        output_data: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProvenanceRecord:
        """
        Create a provenance record for a single pipeline stage.

        Computes SHA-256 hashes of input and output data, then derives
        a stage hash that links the two.

        Args:
            stage: Pipeline stage identifier.
            input_data: Input data to this stage.
            output_data: Output data from this stage.
            metadata: Optional metadata dictionary.

        Returns:
            Immutable ProvenanceRecord with computed hashes.
        """
        with self._lock:
            stage_str = stage.value if isinstance(stage, PipelineStage) else str(stage)
            input_hash = _compute_hash(input_data)
            output_hash = _compute_hash(output_data)
            stage_hash = _compute_stage_hash(stage_str, input_hash, output_hash)

            return ProvenanceRecord(
                record_id=str(uuid.uuid4()),
                stage=stage_str,
                input_hash=input_hash,
                output_hash=output_hash,
                stage_hash=stage_hash,
                timestamp=datetime.now(timezone.utc).isoformat(),
                agent_id=self.agent_id,
                agent_version=self.agent_version,
                metadata=metadata or {},
            )

    def build_full_chain(
        self, records: List[ProvenanceRecord]
    ) -> ProvenanceChain:
        """
        Build a complete provenance chain from a list of records.

        Creates a chain containing all provided records and computes
        the chain-level hash from the ordered stage hashes.

        Args:
            records: Ordered list of provenance records (one per stage).

        Returns:
            ProvenanceChain with computed chain_hash.
        """
        with self._lock:
            stage_hashes = [r.stage_hash for r in records]
            chain_hash = _compute_chain_hash(stage_hashes)

            return ProvenanceChain(
                chain_id=str(uuid.uuid4()),
                records=list(records),
                chain_hash=chain_hash,
                created_at=datetime.now(timezone.utc).isoformat(),
                agent_id=self.agent_id,
            )

    def verify_chain(self, chain: ProvenanceChain) -> bool:
        """
        Verify integrity of a provenance chain.

        Checks:
        1. Each record's stage_hash matches recomputed value.
        2. The chain_hash matches recomputation from stage hashes.

        Args:
            chain: ProvenanceChain to verify.

        Returns:
            True if chain is valid, False otherwise.
        """
        with self._lock:
            if not chain.records:
                return True

            # Verify each record's stage_hash
            for record in chain.records:
                expected_stage_hash = _compute_stage_hash(
                    record.stage,
                    record.input_hash,
                    record.output_hash,
                )
                if record.stage_hash != expected_stage_hash:
                    logger.warning(
                        "Stage hash mismatch for record %s stage %s",
                        record.record_id,
                        record.stage,
                    )
                    return False

            # Verify the chain hash
            stage_hashes = [r.stage_hash for r in chain.records]
            expected_chain_hash = _compute_chain_hash(stage_hashes)
            if chain.chain_hash != expected_chain_hash:
                logger.warning(
                    "Chain hash mismatch for chain %s",
                    chain.chain_id,
                )
                return False

            return True

    def _serialize(self, data: Any) -> str:
        """
        Deterministic JSON serialization (instance method wrapper).

        Args:
            data: Data to serialize.

        Returns:
            Deterministic JSON string.
        """
        return _serialize(data)


# ============================================================================
# STANDALONE HASH FUNCTIONS (10)
# ============================================================================


def hash_raw_input(raw_record: Dict[str, Any]) -> str:
    """
    Hash raw input record (Stage P1).

    Args:
        raw_record: Raw organizational data record.

    Returns:
        SHA-256 hash.
    """
    return _compute_hash(raw_record)


def hash_code_lookup(
    code_type: str,
    code_value: str,
    lookup_result: Dict[str, Any],
) -> str:
    """
    Hash industry code lookup result (Stage P2).

    Args:
        code_type: Code system (naics, isic, unspsc, hs_code).
        code_value: Code value looked up.
        lookup_result: Lookup result with category mappings.

    Returns:
        SHA-256 hash.
    """
    data = {
        "code_type": code_type,
        "code_value": code_value,
        "lookup_result": lookup_result,
    }
    return _compute_hash(data)


def hash_classification(
    category: int,
    method: str,
    confidence: float,
) -> str:
    """
    Hash classification decision (Stage P3).

    Args:
        category: Scope 3 category number (1-15).
        method: Classification method used.
        confidence: Classification confidence score (0.0-1.0).

    Returns:
        SHA-256 hash.
    """
    data = {
        "category": category,
        "method": method,
        "confidence": str(confidence),
    }
    return _compute_hash(data)


def hash_boundary(
    direction: str,
    consolidation_approach: str,
    determination_details: Dict[str, Any],
) -> str:
    """
    Hash boundary determination result (Stage P4).

    Args:
        direction: Value chain direction (upstream/downstream).
        consolidation_approach: Consolidation approach used.
        determination_details: Additional determination details.

    Returns:
        SHA-256 hash.
    """
    data = {
        "direction": direction,
        "consolidation_approach": consolidation_approach,
        "details": determination_details,
    }
    return _compute_hash(data)


def hash_double_counting(
    rules_checked: List[str],
    overlaps_found: List[Dict[str, Any]],
) -> str:
    """
    Hash double-counting check result (Stage P5).

    Args:
        rules_checked: DC-SCM rule IDs that were evaluated.
        overlaps_found: List of overlaps detected.

    Returns:
        SHA-256 hash.
    """
    data = {
        "rules_checked": sorted(rules_checked),
        "overlaps_found": overlaps_found,
    }
    return _compute_hash(data)


def hash_split_allocation(
    splits: List[Dict[str, Any]],
) -> str:
    """
    Hash multi-category split ratios (Stage P6).

    Args:
        splits: List of split allocations with category and ratio.

    Returns:
        SHA-256 hash.
    """
    return _compute_hash({"splits": splits})


def hash_approach(
    approach: str,
    data_quality_tier: int,
    rationale: str,
) -> str:
    """
    Hash calculation approach recommendation (Stage P7).

    Args:
        approach: Recommended calculation approach.
        data_quality_tier: Data quality tier (1-5).
        rationale: Explanation for the recommendation.

    Returns:
        SHA-256 hash.
    """
    data = {
        "approach": approach,
        "data_quality_tier": data_quality_tier,
        "rationale": rationale,
    }
    return _compute_hash(data)


def hash_routing(
    target_agent: str,
    category: int,
    transformed_input: Dict[str, Any],
) -> str:
    """
    Hash routing instruction (Stage P8).

    Args:
        target_agent: Target agent identifier.
        category: Target Scope 3 category number.
        transformed_input: Transformed input for the target agent.

    Returns:
        SHA-256 hash.
    """
    data = {
        "target_agent": target_agent,
        "category": category,
        "transformed_input": transformed_input,
    }
    return _compute_hash(data)


def hash_completeness(
    categories_present: List[int],
    categories_missing: List[int],
    coverage_pct: float,
) -> str:
    """
    Hash completeness screening result (Stage P9).

    Args:
        categories_present: Categories with data.
        categories_missing: Categories without data.
        coverage_pct: Coverage percentage.

    Returns:
        SHA-256 hash.
    """
    data = {
        "categories_present": sorted(categories_present),
        "categories_missing": sorted(categories_missing),
        "coverage_pct": str(coverage_pct),
    }
    return _compute_hash(data)


def build_chain_hash(stage_hashes: List[str]) -> str:
    """
    Build final chain hash from all stage hashes (Stage P10).

    Chain hash = SHA-256(P1 || P2 || ... || P9)

    Args:
        stage_hashes: Ordered list of per-stage hashes.

    Returns:
        Final SHA-256 chain hash.
    """
    return _compute_chain_hash(stage_hashes)


# ============================================================================
# SINGLETON ACCESS
# ============================================================================

_engine_instance: Optional[ProvenanceEngine] = None
_engine_lock = threading.RLock()


def get_provenance_engine() -> ProvenanceEngine:
    """
    Get singleton ProvenanceEngine instance.

    Thread-safe singleton pattern.

    Returns:
        ProvenanceEngine instance.
    """
    global _engine_instance
    with _engine_lock:
        if _engine_instance is None:
            _engine_instance = ProvenanceEngine()
        return _engine_instance


def reset_provenance_engine() -> None:
    """
    Reset singleton ProvenanceEngine instance.

    Useful for testing. Creates a fresh instance on next access.
    """
    global _engine_instance
    with _engine_lock:
        _engine_instance = None


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "PipelineStage",
    # Models
    "ProvenanceRecord",
    "ProvenanceChain",
    # Engine
    "ProvenanceEngine",
    # Singleton
    "get_provenance_engine",
    "reset_provenance_engine",
    # Hash functions (10)
    "hash_raw_input",
    "hash_code_lookup",
    "hash_classification",
    "hash_boundary",
    "hash_double_counting",
    "hash_split_allocation",
    "hash_approach",
    "hash_routing",
    "hash_completeness",
    "build_chain_hash",
    # Hash utilities
    "_serialize",
    "_compute_hash",
    "_compute_stage_hash",
    "_compute_chain_hash",
]
