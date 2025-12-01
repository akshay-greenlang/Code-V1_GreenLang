"""
GL-013 PREDICTMAINT - Provenance and Audit Trail Module

This module provides comprehensive audit trail capabilities for all
predictive maintenance calculations, ensuring regulatory compliance
and zero-hallucination verification.

Key Features:
- SHA-256 hashing for all calculations
- Merkle tree for calculation chains
- Immutable record storage
- Calculation replay capability
- Complete provenance tracking

Reference Standards:
- NIST SP 800-185 (SHA-3 Standard)
- ISO/IEC 27001 (Information Security)
- SOC 2 Type II Compliance

Author: GL-CalculatorEngineer
Version: 1.0.0
License: Proprietary - GreenLang
"""

from decimal import Decimal
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timezone
from enum import Enum, auto
import hashlib
import json
import uuid
from abc import ABC, abstractmethod


# =============================================================================
# PROVENANCE ENUMS
# =============================================================================

class CalculationType(Enum):
    """Types of calculations tracked."""
    RUL_WEIBULL = auto()
    RUL_EXPONENTIAL = auto()
    RUL_LOGNORMAL = auto()
    FAILURE_PROBABILITY = auto()
    HAZARD_RATE = auto()
    VIBRATION_ANALYSIS = auto()
    THERMAL_DEGRADATION = auto()
    MAINTENANCE_OPTIMIZATION = auto()
    SPARE_PARTS = auto()
    ANOMALY_DETECTION = auto()
    UNIT_CONVERSION = auto()


class ProvenanceStatus(Enum):
    """Status of provenance record."""
    PENDING = auto()
    VALIDATED = auto()
    INVALIDATED = auto()
    ARCHIVED = auto()


# =============================================================================
# CALCULATION STEP RECORD
# =============================================================================

@dataclass(frozen=True)
class CalculationStep:
    """
    Immutable record of a single calculation step.

    Attributes:
        step_number: Sequential step number
        operation: Mathematical operation performed
        description: Human-readable description
        inputs: Dictionary of input values
        output_name: Name of output variable
        output_value: Result value
        formula: Mathematical formula used
        reference: Academic/standard reference
    """
    step_number: int
    operation: str
    description: str
    inputs: Dict[str, Any]
    output_name: str
    output_value: Any
    formula: str = ""
    reference: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "step_number": self.step_number,
            "operation": self.operation,
            "description": self.description,
            "inputs": self._serialize_inputs(self.inputs),
            "output_name": self.output_name,
            "output_value": self._serialize_value(self.output_value),
            "formula": self.formula,
            "reference": self.reference,
        }

    @staticmethod
    def _serialize_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize input values to JSON-compatible format."""
        result = {}
        for key, value in inputs.items():
            result[key] = CalculationStep._serialize_value(value)
        return result

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        """Serialize a single value."""
        if isinstance(value, Decimal):
            return str(value)
        elif isinstance(value, (list, tuple)):
            return [CalculationStep._serialize_value(v) for v in value]
        elif isinstance(value, dict):
            return {k: CalculationStep._serialize_value(v) for k, v in value.items()}
        return value

    def calculate_hash(self) -> str:
        """Calculate SHA-256 hash of this step."""
        step_data = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(step_data.encode("utf-8")).hexdigest()


# =============================================================================
# PROVENANCE RECORD
# =============================================================================

@dataclass(frozen=True)
class ProvenanceRecord:
    """
    Immutable provenance record for a complete calculation.

    This record provides:
    - Complete input/output traceability
    - SHA-256 hash for tamper detection
    - Calculation replay capability
    - Regulatory compliance documentation

    Attributes:
        record_id: Unique identifier (UUID)
        calculation_type: Type of calculation
        timestamp: UTC timestamp of calculation
        inputs: All input parameters
        outputs: All output values
        steps: List of calculation steps
        final_hash: SHA-256 of complete record
        metadata: Additional context
    """
    record_id: str
    calculation_type: CalculationType
    timestamp: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    steps: Tuple[CalculationStep, ...]
    final_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "record_id": self.record_id,
            "calculation_type": self.calculation_type.name,
            "timestamp": self.timestamp,
            "inputs": self._serialize_dict(self.inputs),
            "outputs": self._serialize_dict(self.outputs),
            "steps": [step.to_dict() for step in self.steps],
            "final_hash": self.final_hash,
            "metadata": self.metadata,
        }

    @staticmethod
    def _serialize_dict(d: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize dictionary values."""
        result = {}
        for key, value in d.items():
            if isinstance(value, Decimal):
                result[key] = str(value)
            elif isinstance(value, Enum):
                result[key] = value.name
            elif isinstance(value, dict):
                result[key] = ProvenanceRecord._serialize_dict(value)
            elif isinstance(value, (list, tuple)):
                result[key] = [
                    str(v) if isinstance(v, Decimal) else v
                    for v in value
                ]
            else:
                result[key] = value
        return result

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def verify_integrity(self) -> bool:
        """
        Verify the integrity of this provenance record.

        Returns:
            True if the record has not been tampered with
        """
        recalculated_hash = ProvenanceBuilder._calculate_final_hash(
            self.record_id,
            self.calculation_type,
            self.timestamp,
            self.inputs,
            self.outputs,
            self.steps
        )
        return recalculated_hash == self.final_hash


# =============================================================================
# MERKLE TREE NODE
# =============================================================================

@dataclass(frozen=True)
class MerkleNode:
    """
    Node in a Merkle tree for calculation chain verification.

    Merkle trees allow efficient verification of calculation sequences
    without needing to recompute all individual hashes.
    """
    node_hash: str
    left_hash: Optional[str] = None
    right_hash: Optional[str] = None
    is_leaf: bool = True
    data_hash: Optional[str] = None  # Original data hash for leaf nodes


class MerkleTree:
    """
    Merkle tree for verifiable calculation chains.

    A Merkle tree provides:
    - O(log n) verification of any single calculation
    - Tamper detection across calculation sequences
    - Efficient proof of inclusion

    Reference: Merkle, R. (1988). "A Digital Signature Based on a
    Conventional Encryption Function". Crypto '87.
    """

    def __init__(self, hash_function=hashlib.sha256):
        """
        Initialize Merkle tree.

        Args:
            hash_function: Hash function to use (default: SHA-256)
        """
        self._hash_function = hash_function
        self._leaves: List[str] = []
        self._root: Optional[MerkleNode] = None
        self._nodes: Dict[str, MerkleNode] = {}

    def add_calculation(self, provenance_record: ProvenanceRecord) -> str:
        """
        Add a calculation to the Merkle tree.

        Args:
            provenance_record: The provenance record to add

        Returns:
            The leaf hash for this calculation
        """
        leaf_hash = provenance_record.final_hash
        self._leaves.append(leaf_hash)
        self._rebuild_tree()
        return leaf_hash

    def add_hash(self, data_hash: str) -> str:
        """
        Add a pre-computed hash to the tree.

        Args:
            data_hash: SHA-256 hash to add

        Returns:
            The leaf hash
        """
        self._leaves.append(data_hash)
        self._rebuild_tree()
        return data_hash

    def get_root_hash(self) -> Optional[str]:
        """Get the root hash of the tree."""
        return self._root.node_hash if self._root else None

    def get_proof(self, leaf_hash: str) -> List[Tuple[str, str]]:
        """
        Get Merkle proof for a leaf.

        The proof consists of sibling hashes needed to reconstruct
        the path from the leaf to the root.

        Args:
            leaf_hash: The leaf hash to prove

        Returns:
            List of (sibling_hash, position) tuples
            position is "left" or "right"
        """
        if leaf_hash not in self._leaves:
            raise ValueError(f"Leaf hash not found: {leaf_hash}")

        proof = []
        index = self._leaves.index(leaf_hash)
        current_level = self._leaves.copy()

        while len(current_level) > 1:
            if index % 2 == 0:
                # We are the left node
                sibling_index = index + 1
                position = "right"
            else:
                # We are the right node
                sibling_index = index - 1
                position = "left"

            if sibling_index < len(current_level):
                proof.append((current_level[sibling_index], position))
            else:
                # Odd number of nodes, duplicate the last one
                proof.append((current_level[index], position))

            # Move to next level
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                combined = self._hash_pair(left, right)
                next_level.append(combined)

            current_level = next_level
            index = index // 2

        return proof

    def verify_proof(
        self,
        leaf_hash: str,
        proof: List[Tuple[str, str]],
        root_hash: str
    ) -> bool:
        """
        Verify a Merkle proof.

        Args:
            leaf_hash: The hash of the leaf to verify
            proof: The proof from get_proof()
            root_hash: The expected root hash

        Returns:
            True if the proof is valid
        """
        current_hash = leaf_hash

        for sibling_hash, position in proof:
            if position == "left":
                current_hash = self._hash_pair(sibling_hash, current_hash)
            else:
                current_hash = self._hash_pair(current_hash, sibling_hash)

        return current_hash == root_hash

    def _rebuild_tree(self) -> None:
        """Rebuild the Merkle tree from leaves."""
        if not self._leaves:
            self._root = None
            return

        current_level = self._leaves.copy()

        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                combined = self._hash_pair(left, right)
                next_level.append(combined)

                # Store node
                node = MerkleNode(
                    node_hash=combined,
                    left_hash=left,
                    right_hash=right,
                    is_leaf=False
                )
                self._nodes[combined] = node

            current_level = next_level

        # Create root node
        self._root = MerkleNode(
            node_hash=current_level[0],
            is_leaf=len(self._leaves) == 1
        )

    def _hash_pair(self, left: str, right: str) -> str:
        """Hash two values together."""
        combined = f"{left}{right}".encode("utf-8")
        return self._hash_function(combined).hexdigest()

    def get_leaf_count(self) -> int:
        """Get the number of leaves in the tree."""
        return len(self._leaves)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize tree state."""
        return {
            "root_hash": self.get_root_hash(),
            "leaf_count": self.get_leaf_count(),
            "leaves": self._leaves.copy(),
        }


# =============================================================================
# PROVENANCE BUILDER
# =============================================================================

class ProvenanceBuilder:
    """
    Builder for creating provenance records.

    This builder ensures all calculations are properly documented
    with complete audit trails.

    Example:
        >>> builder = ProvenanceBuilder(CalculationType.RUL_WEIBULL)
        >>> builder.add_input("operating_hours", Decimal("5000"))
        >>> builder.add_step(1, "lookup", "Get Weibull parameters", {...}, "beta", Decimal("2.5"))
        >>> builder.add_output("rul_hours", Decimal("3000"))
        >>> record = builder.build()
        >>> print(record.final_hash)
    """

    def __init__(
        self,
        calculation_type: CalculationType,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize provenance builder.

        Args:
            calculation_type: Type of calculation being documented
            metadata: Optional metadata dictionary
        """
        self._record_id = str(uuid.uuid4())
        self._calculation_type = calculation_type
        self._timestamp = datetime.now(timezone.utc).isoformat()
        self._inputs: Dict[str, Any] = {}
        self._outputs: Dict[str, Any] = {}
        self._steps: List[CalculationStep] = []
        self._metadata = metadata or {}

    def add_input(self, name: str, value: Any) -> "ProvenanceBuilder":
        """
        Add an input parameter.

        Args:
            name: Parameter name
            value: Parameter value

        Returns:
            Self for method chaining
        """
        self._inputs[name] = value
        return self

    def add_inputs(self, inputs: Dict[str, Any]) -> "ProvenanceBuilder":
        """
        Add multiple input parameters.

        Args:
            inputs: Dictionary of parameters

        Returns:
            Self for method chaining
        """
        self._inputs.update(inputs)
        return self

    def add_output(self, name: str, value: Any) -> "ProvenanceBuilder":
        """
        Add an output value.

        Args:
            name: Output name
            value: Output value

        Returns:
            Self for method chaining
        """
        self._outputs[name] = value
        return self

    def add_outputs(self, outputs: Dict[str, Any]) -> "ProvenanceBuilder":
        """
        Add multiple output values.

        Args:
            outputs: Dictionary of outputs

        Returns:
            Self for method chaining
        """
        self._outputs.update(outputs)
        return self

    def add_step(
        self,
        step_number: int,
        operation: str,
        description: str,
        inputs: Dict[str, Any],
        output_name: str,
        output_value: Any,
        formula: str = "",
        reference: str = ""
    ) -> "ProvenanceBuilder":
        """
        Add a calculation step.

        Args:
            step_number: Sequential step number
            operation: Operation type (e.g., "multiply", "lookup")
            description: Human-readable description
            inputs: Step input values
            output_name: Name of output variable
            output_value: Step result
            formula: Mathematical formula
            reference: Academic/standard reference

        Returns:
            Self for method chaining
        """
        step = CalculationStep(
            step_number=step_number,
            operation=operation,
            description=description,
            inputs=inputs,
            output_name=output_name,
            output_value=output_value,
            formula=formula,
            reference=reference
        )
        self._steps.append(step)
        return self

    def add_metadata(self, key: str, value: Any) -> "ProvenanceBuilder":
        """
        Add metadata.

        Args:
            key: Metadata key
            value: Metadata value

        Returns:
            Self for method chaining
        """
        self._metadata[key] = value
        return self

    def build(self) -> ProvenanceRecord:
        """
        Build the immutable provenance record.

        Returns:
            Complete provenance record with hash
        """
        # Calculate final hash
        final_hash = self._calculate_final_hash(
            self._record_id,
            self._calculation_type,
            self._timestamp,
            self._inputs,
            self._outputs,
            self._steps
        )

        return ProvenanceRecord(
            record_id=self._record_id,
            calculation_type=self._calculation_type,
            timestamp=self._timestamp,
            inputs=self._inputs.copy(),
            outputs=self._outputs.copy(),
            steps=tuple(self._steps),
            final_hash=final_hash,
            metadata=self._metadata.copy()
        )

    @staticmethod
    def _calculate_final_hash(
        record_id: str,
        calculation_type: CalculationType,
        timestamp: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        steps: List[CalculationStep]
    ) -> str:
        """
        Calculate the final SHA-256 hash for the record.

        This hash uniquely identifies the complete calculation
        and enables tamper detection.
        """
        # Serialize all components
        hash_data = {
            "record_id": record_id,
            "calculation_type": calculation_type.name,
            "timestamp": timestamp,
            "inputs": ProvenanceRecord._serialize_dict(inputs),
            "outputs": ProvenanceRecord._serialize_dict(outputs),
            "steps": [step.to_dict() for step in steps],
        }

        # Create deterministic JSON (sorted keys)
        json_str = json.dumps(hash_data, sort_keys=True, default=str)

        # Calculate SHA-256
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


# =============================================================================
# PROVENANCE STORE (ABSTRACT)
# =============================================================================

class ProvenanceStore(ABC):
    """
    Abstract base class for provenance storage.

    Implementations can use various backends:
    - In-memory (for testing)
    - File-based (for local storage)
    - Database (for production)
    - Blockchain (for highest trust)
    """

    @abstractmethod
    def store(self, record: ProvenanceRecord) -> str:
        """
        Store a provenance record.

        Args:
            record: The record to store

        Returns:
            Record ID
        """
        pass

    @abstractmethod
    def retrieve(self, record_id: str) -> Optional[ProvenanceRecord]:
        """
        Retrieve a provenance record.

        Args:
            record_id: The record ID

        Returns:
            The record or None if not found
        """
        pass

    @abstractmethod
    def verify(self, record_id: str) -> bool:
        """
        Verify the integrity of a stored record.

        Args:
            record_id: The record ID

        Returns:
            True if the record is valid
        """
        pass

    @abstractmethod
    def list_records(
        self,
        calculation_type: Optional[CalculationType] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> List[str]:
        """
        List record IDs matching criteria.

        Args:
            calculation_type: Filter by calculation type
            start_time: Start of time range (ISO format)
            end_time: End of time range (ISO format)

        Returns:
            List of matching record IDs
        """
        pass


# =============================================================================
# IN-MEMORY PROVENANCE STORE
# =============================================================================

class InMemoryProvenanceStore(ProvenanceStore):
    """
    In-memory implementation of provenance store.

    Suitable for testing and development. For production,
    use a persistent implementation.
    """

    def __init__(self):
        """Initialize the store."""
        self._records: Dict[str, ProvenanceRecord] = {}
        self._merkle_tree = MerkleTree()

    def store(self, record: ProvenanceRecord) -> str:
        """Store a provenance record."""
        self._records[record.record_id] = record
        self._merkle_tree.add_calculation(record)
        return record.record_id

    def retrieve(self, record_id: str) -> Optional[ProvenanceRecord]:
        """Retrieve a provenance record."""
        return self._records.get(record_id)

    def verify(self, record_id: str) -> bool:
        """Verify the integrity of a stored record."""
        record = self._records.get(record_id)
        if record is None:
            return False
        return record.verify_integrity()

    def list_records(
        self,
        calculation_type: Optional[CalculationType] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> List[str]:
        """List record IDs matching criteria."""
        results = []
        for record_id, record in self._records.items():
            # Filter by calculation type
            if calculation_type and record.calculation_type != calculation_type:
                continue

            # Filter by time range
            if start_time and record.timestamp < start_time:
                continue
            if end_time and record.timestamp > end_time:
                continue

            results.append(record_id)

        return results

    def get_merkle_root(self) -> Optional[str]:
        """Get the current Merkle tree root hash."""
        return self._merkle_tree.get_root_hash()

    def get_merkle_proof(self, record_id: str) -> List[Tuple[str, str]]:
        """Get Merkle proof for a record."""
        record = self._records.get(record_id)
        if record is None:
            raise ValueError(f"Record not found: {record_id}")
        return self._merkle_tree.get_proof(record.final_hash)

    def verify_merkle_proof(
        self,
        record_id: str,
        proof: List[Tuple[str, str]],
        root_hash: str
    ) -> bool:
        """Verify a Merkle proof for a record."""
        record = self._records.get(record_id)
        if record is None:
            return False
        return self._merkle_tree.verify_proof(record.final_hash, proof, root_hash)

    def get_record_count(self) -> int:
        """Get the total number of stored records."""
        return len(self._records)

    def clear(self) -> None:
        """Clear all records (for testing)."""
        self._records.clear()
        self._merkle_tree = MerkleTree()


# =============================================================================
# CALCULATION REPLAY
# =============================================================================

class CalculationReplayer:
    """
    Replay calculations from provenance records for verification.

    This class enables:
    - Independent verification of past calculations
    - Audit trail validation
    - Debugging and analysis

    Example:
        >>> replayer = CalculationReplayer(store)
        >>> result = replayer.replay(record_id)
        >>> assert result.matches_original == True
    """

    @dataclass(frozen=True)
    class ReplayResult:
        """Result of a calculation replay."""
        record_id: str
        original_outputs: Dict[str, Any]
        replayed_outputs: Dict[str, Any]
        matches_original: bool
        discrepancies: Dict[str, Tuple[Any, Any]]  # {name: (original, replayed)}
        replay_time_ms: float

    def __init__(self, store: ProvenanceStore):
        """
        Initialize the replayer.

        Args:
            store: Provenance store containing records
        """
        self._store = store
        self._calculators: Dict[CalculationType, Any] = {}

    def register_calculator(
        self,
        calculation_type: CalculationType,
        calculator: Any
    ) -> None:
        """
        Register a calculator for replay.

        Args:
            calculation_type: The type of calculation
            calculator: Calculator instance with compatible interface
        """
        self._calculators[calculation_type] = calculator

    def replay(self, record_id: str) -> "CalculationReplayer.ReplayResult":
        """
        Replay a calculation and compare results.

        Args:
            record_id: The record to replay

        Returns:
            ReplayResult with comparison
        """
        import time
        start_time = time.perf_counter()

        # Retrieve record
        record = self._store.retrieve(record_id)
        if record is None:
            raise ValueError(f"Record not found: {record_id}")

        # Get calculator
        calculator = self._calculators.get(record.calculation_type)
        if calculator is None:
            raise ValueError(
                f"No calculator registered for {record.calculation_type.name}"
            )

        # Replay calculation
        try:
            replayed_outputs = calculator.calculate(**record.inputs)
            if hasattr(replayed_outputs, "to_dict"):
                replayed_outputs = replayed_outputs.to_dict()
            elif hasattr(replayed_outputs, "__dict__"):
                replayed_outputs = {
                    k: v for k, v in replayed_outputs.__dict__.items()
                    if not k.startswith("_")
                }
        except Exception as e:
            replayed_outputs = {"error": str(e)}

        # Compare outputs
        discrepancies = {}
        original = record.outputs
        matches = True

        for key in set(original.keys()) | set(replayed_outputs.keys()):
            orig_val = original.get(key)
            replay_val = replayed_outputs.get(key)

            # Compare values (handle Decimal comparison)
            if not self._values_equal(orig_val, replay_val):
                discrepancies[key] = (orig_val, replay_val)
                matches = False

        end_time = time.perf_counter()
        replay_time_ms = (end_time - start_time) * 1000

        return self.ReplayResult(
            record_id=record_id,
            original_outputs=original,
            replayed_outputs=replayed_outputs,
            matches_original=matches,
            discrepancies=discrepancies,
            replay_time_ms=replay_time_ms
        )

    @staticmethod
    def _values_equal(val1: Any, val2: Any) -> bool:
        """Compare two values for equality."""
        # Handle None
        if val1 is None and val2 is None:
            return True
        if val1 is None or val2 is None:
            return False

        # Handle Decimal
        if isinstance(val1, Decimal) and isinstance(val2, Decimal):
            return val1 == val2
        if isinstance(val1, Decimal):
            try:
                return val1 == Decimal(str(val2))
            except:
                return False
        if isinstance(val2, Decimal):
            try:
                return Decimal(str(val1)) == val2
            except:
                return False

        # Handle strings that might be Decimal
        if isinstance(val1, str) and isinstance(val2, str):
            try:
                return Decimal(val1) == Decimal(val2)
            except:
                return val1 == val2

        return val1 == val2


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_hash(data: Any) -> str:
    """
    Calculate SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (will be JSON serialized)

    Returns:
        SHA-256 hash as hex string
    """
    if isinstance(data, str):
        json_str = data
    else:
        json_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


def verify_hash(data: Any, expected_hash: str) -> bool:
    """
    Verify that data matches an expected hash.

    Args:
        data: Data to verify
        expected_hash: Expected SHA-256 hash

    Returns:
        True if hash matches
    """
    actual_hash = calculate_hash(data)
    return actual_hash == expected_hash


def create_provenance_record(
    calculation_type: CalculationType,
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    steps: Optional[List[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> ProvenanceRecord:
    """
    Convenience function to create a provenance record.

    Args:
        calculation_type: Type of calculation
        inputs: Input parameters
        outputs: Output values
        steps: Optional list of step dictionaries
        metadata: Optional metadata

    Returns:
        Complete provenance record
    """
    builder = ProvenanceBuilder(calculation_type, metadata)
    builder.add_inputs(inputs)
    builder.add_outputs(outputs)

    if steps:
        for step_dict in steps:
            builder.add_step(**step_dict)

    return builder.build()


# =============================================================================
# GLOBAL STORE INSTANCE
# =============================================================================

# Default global provenance store
_global_store: Optional[ProvenanceStore] = None


def get_global_store() -> ProvenanceStore:
    """Get the global provenance store."""
    global _global_store
    if _global_store is None:
        _global_store = InMemoryProvenanceStore()
    return _global_store


def set_global_store(store: ProvenanceStore) -> None:
    """Set the global provenance store."""
    global _global_store
    _global_store = store


def store_provenance(record: ProvenanceRecord) -> str:
    """Store a provenance record in the global store."""
    return get_global_store().store(record)


def retrieve_provenance(record_id: str) -> Optional[ProvenanceRecord]:
    """Retrieve a provenance record from the global store."""
    return get_global_store().retrieve(record_id)


def verify_provenance(record_id: str) -> bool:
    """Verify a provenance record in the global store."""
    return get_global_store().verify(record_id)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "CalculationType",
    "ProvenanceStatus",

    # Data classes
    "CalculationStep",
    "ProvenanceRecord",
    "MerkleNode",

    # Classes
    "MerkleTree",
    "ProvenanceBuilder",
    "ProvenanceStore",
    "InMemoryProvenanceStore",
    "CalculationReplayer",

    # Functions
    "calculate_hash",
    "verify_hash",
    "create_provenance_record",
    "get_global_store",
    "set_global_store",
    "store_provenance",
    "retrieve_provenance",
    "verify_provenance",
]
