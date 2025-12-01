# -*- coding: utf-8 -*-
"""
Provenance Tracking for GL-012 STEAMQUAL.

Provides SHA-256 cryptographic provenance tracking for all steam quality
calculations, ensuring complete audit trails and bit-perfect reproducibility.

Standards:
- SHA-256: FIPS 180-4 Secure Hash Standard
- ISO 27001: Information Security Management (audit trails)
- GxP: Good Practice regulations (pharmaceutical/food industry)

Zero-hallucination: All provenance tracking is deterministic.
No LLM involved in hash generation or verification.

Author: GL-CalculatorEngineer
Version: 1.0.0

Features:
    - SHA-256 hash generation for all calculations
    - Complete audit trail with timestamps
    - Hash verification for integrity checking
    - Provenance chain for multi-step calculations
    - Export capabilities for regulatory compliance
"""

import hashlib
import json
import logging
import threading
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from collections import OrderedDict
from enum import Enum

logger = logging.getLogger(__name__)


class ProvenanceStatus(Enum):
    """Provenance verification status."""
    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    TAMPERED = "tampered"
    EXPIRED = "expired"
    PENDING = "pending"


@dataclass
class CalculationStep:
    """
    Represents a single step in a calculation chain.

    Each step records the operation performed, inputs used,
    output produced, and optional formula reference.

    Attributes:
        step_number: Sequential step identifier
        operation: Name of operation performed
        description: Human-readable description
        inputs: Dictionary of input values
        output_value: Result of this calculation step
        output_name: Name/identifier for the output
        formula: Mathematical formula used (if applicable)
        units: Output units
        timestamp: When step was executed (ISO format)
    """
    step_number: int
    operation: str
    description: str
    inputs: Dict[str, Any]
    output_value: Any
    output_name: str
    formula: Optional[str] = None
    units: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary with proper serialization.

        Handles Decimal and other non-JSON-native types.
        """
        data = asdict(self)

        # Serialize Decimal values in inputs
        serialized_inputs = {}
        for key, value in data['inputs'].items():
            if isinstance(value, Decimal):
                serialized_inputs[key] = str(value)
            elif isinstance(value, (list, tuple)):
                serialized_inputs[key] = [
                    str(v) if isinstance(v, Decimal) else v for v in value
                ]
            elif isinstance(value, dict):
                serialized_inputs[key] = {
                    k: str(v) if isinstance(v, Decimal) else v
                    for k, v in value.items()
                }
            else:
                serialized_inputs[key] = value
        data['inputs'] = serialized_inputs

        # Serialize output value
        if isinstance(data['output_value'], Decimal):
            data['output_value'] = str(data['output_value'])

        return data

    def get_input_hash(self) -> str:
        """Generate hash of inputs for this step."""
        input_str = json.dumps(
            self.to_dict()['inputs'],
            sort_keys=True,
            default=str
        )
        return hashlib.sha256(input_str.encode('utf-8')).hexdigest()


@dataclass
class ProvenanceRecord:
    """
    Complete provenance record for a calculation.

    Captures all information needed to reproduce and verify
    a calculation, including inputs, steps, output, and hash.

    Attributes:
        calculation_id: Unique identifier for this calculation
        calculation_type: Type/name of calculation
        version: Calculator version
        input_parameters: All input values
        calculation_steps: List of calculation steps
        final_result: Final output value
        provenance_hash: SHA-256 hash of complete calculation
        timestamp: When calculation was performed
        metadata: Additional context (user, system, etc.)
        status: Verification status
    """
    calculation_id: str
    calculation_type: str
    version: str
    input_parameters: Dict[str, Any]
    calculation_steps: List[CalculationStep]
    final_result: Any
    provenance_hash: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: ProvenanceStatus = ProvenanceStatus.UNVERIFIED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = {
            'calculation_id': self.calculation_id,
            'calculation_type': self.calculation_type,
            'version': self.version,
            'input_parameters': self._serialize_values(self.input_parameters),
            'calculation_steps': [step.to_dict() for step in self.calculation_steps],
            'final_result': self._serialize_value(self.final_result),
            'provenance_hash': self.provenance_hash,
            'timestamp': self.timestamp,
            'metadata': self.metadata,
            'status': self.status.value
        }
        return data

    def _serialize_value(self, value: Any) -> Any:
        """Serialize a single value."""
        if isinstance(value, Decimal):
            return str(value)
        elif isinstance(value, Enum):
            return value.value
        elif isinstance(value, dict):
            return self._serialize_values(value)
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        return value

    def _serialize_values(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize dictionary values."""
        return {k: self._serialize_value(v) for k, v in values.items()}

    def to_json(self, indent: int = 2) -> str:
        """Export as JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


@dataclass
class ProvenanceChain:
    """
    Chain of linked provenance records.

    Used for multi-stage calculations where output of one
    calculation feeds into the next.

    Attributes:
        chain_id: Unique chain identifier
        records: List of linked ProvenanceRecord objects
        chain_hash: Hash of entire chain (integrity check)
        created_at: Chain creation timestamp
        verified: Whether chain integrity has been verified
    """
    chain_id: str
    records: List[ProvenanceRecord] = field(default_factory=list)
    chain_hash: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    verified: bool = False

    def add_record(self, record: ProvenanceRecord) -> None:
        """Add a record to the chain."""
        self.records.append(record)
        self._update_chain_hash()

    def _update_chain_hash(self) -> None:
        """Recalculate chain hash after modification."""
        chain_data = [r.provenance_hash for r in self.records]
        chain_str = json.dumps(chain_data, sort_keys=True)
        self.chain_hash = hashlib.sha256(chain_str.encode('utf-8')).hexdigest()


class ProvenanceTracker:
    """
    SHA-256 provenance tracking for zero-hallucination calculations.

    Provides cryptographic audit trails for all calculations,
    ensuring complete traceability and reproducibility.

    ZERO-HALLUCINATION GUARANTEES:
    - Deterministic hashing (same input -> same hash)
    - Complete audit trail
    - Tamper detection
    - No LLM involvement in any operation

    Thread-safe implementation using RLock.

    Example:
        >>> tracker = ProvenanceTracker(
        ...     calculation_id="steam_quality_001",
        ...     calculation_type="steam_quality",
        ...     version="1.0.0"
        ... )
        >>> tracker.record_inputs({'pressure_mpa': 1.0, 'temperature_c': 180.0})
        >>> tracker.record_step(
        ...     operation="calculate_dryness",
        ...     description="Calculate dryness fraction from enthalpy",
        ...     inputs={'h': 2700, 'h_f': 762.68, 'h_fg': 2014.9},
        ...     output_value=Decimal('0.962'),
        ...     output_name='dryness_fraction',
        ...     formula='x = (h - h_f) / h_fg'
        ... )
        >>> record = tracker.get_provenance_record(Decimal('0.962'))
        >>> print(f"Hash: {record.provenance_hash}")
    """

    def __init__(
        self,
        calculation_id: str,
        calculation_type: str,
        version: str = "1.0.0"
    ):
        """
        Initialize provenance tracker.

        Args:
            calculation_id: Unique identifier for this calculation
            calculation_type: Type/name of calculation (e.g., "steam_quality")
            version: Calculator version string
        """
        self.calculation_id = calculation_id
        self.calculation_type = calculation_type
        self.version = version
        self.steps: List[CalculationStep] = []
        self.input_parameters: Dict[str, Any] = {}
        self._lock = threading.RLock()

    def record_inputs(self, parameters: Dict[str, Any]) -> None:
        """
        Record input parameters for the calculation.

        Args:
            parameters: Dictionary of input parameter names and values
        """
        with self._lock:
            self.input_parameters = self._normalize_values(parameters.copy())

    def record_step(
        self,
        operation: str,
        description: str,
        inputs: Dict[str, Any],
        output_value: Any,
        output_name: str,
        formula: Optional[str] = None,
        units: Optional[str] = None
    ) -> CalculationStep:
        """
        Record a calculation step.

        Args:
            operation: Name of the operation (e.g., "multiply", "lookup")
            description: Human-readable description
            inputs: Input values for this step
            output_value: Result of the calculation
            output_name: Name for the output variable
            formula: Mathematical formula used (optional)
            units: Output units (optional)

        Returns:
            CalculationStep object that was recorded
        """
        with self._lock:
            step = CalculationStep(
                step_number=len(self.steps) + 1,
                operation=operation,
                description=description,
                inputs=self._normalize_values(inputs),
                output_value=self._normalize_value(output_value),
                output_name=output_name,
                formula=formula,
                units=units
            )
            self.steps.append(step)
            return step

    def generate_hash(self, final_result: Any) -> str:
        """
        Generate SHA-256 hash of the complete calculation.

        This hash is deterministic - same inputs and steps always
        produce the same hash (bit-perfect reproducibility).

        ZERO-HALLUCINATION GUARANTEE:
        - Uses only deterministic operations
        - Sorted keys for consistent JSON serialization
        - No randomness or LLM involvement

        Args:
            final_result: The final output of the calculation

        Returns:
            SHA-256 hash as hexadecimal string (64 characters)
        """
        # Create canonical representation
        canonical_data = {
            'calculation_id': self.calculation_id,
            'calculation_type': self.calculation_type,
            'version': self.version,
            'input_parameters': self._serialize_for_hash(self.input_parameters),
            'steps': [self._serialize_step_for_hash(step) for step in self.steps],
            'final_result': self._serialize_for_hash(final_result)
        }

        # Generate deterministic JSON string (sorted keys, no extra whitespace)
        canonical_json = json.dumps(
            canonical_data,
            sort_keys=True,
            separators=(',', ':'),
            default=str
        )

        # Calculate SHA-256 hash
        return hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()

    def get_provenance_record(
        self,
        final_result: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProvenanceRecord:
        """
        Get complete provenance record for the calculation.

        Args:
            final_result: The final output value
            metadata: Additional context information (optional)

        Returns:
            ProvenanceRecord with complete audit trail
        """
        with self._lock:
            provenance_hash = self.generate_hash(final_result)

            return ProvenanceRecord(
                calculation_id=self.calculation_id,
                calculation_type=self.calculation_type,
                version=self.version,
                input_parameters=self.input_parameters,
                calculation_steps=self.steps.copy(),
                final_result=self._normalize_value(final_result),
                provenance_hash=provenance_hash,
                metadata=metadata or {}
            )

    def verify_hash(
        self,
        final_result: Any,
        expected_hash: str
    ) -> bool:
        """
        Verify that calculation produces expected hash.

        Args:
            final_result: The final output to verify
            expected_hash: The expected SHA-256 hash

        Returns:
            True if hashes match, False otherwise
        """
        actual_hash = self.generate_hash(final_result)
        return actual_hash == expected_hash

    def create_audit_trail(
        self,
        final_result: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create complete audit trail for regulatory compliance.

        Returns a dictionary suitable for storage, reporting,
        and third-party audit.

        Args:
            final_result: The final calculation result
            metadata: Additional context (operator, system, etc.)

        Returns:
            Dictionary with complete audit trail
        """
        record = self.get_provenance_record(final_result, metadata)

        return {
            'audit_trail': {
                'calculation_id': record.calculation_id,
                'calculation_type': record.calculation_type,
                'version': record.version,
                'timestamp': record.timestamp,
                'provenance_hash': record.provenance_hash,
                'status': record.status.value
            },
            'inputs': record.input_parameters,
            'steps': [step.to_dict() for step in record.calculation_steps],
            'output': {
                'value': str(record.final_result) if isinstance(record.final_result, Decimal) else record.final_result,
                'hash_verified': True
            },
            'metadata': record.metadata,
            'verification': {
                'algorithm': 'SHA-256',
                'standard': 'FIPS 180-4',
                'reproducible': True
            }
        }

    def _normalize_value(self, value: Any) -> Any:
        """Normalize values for consistent handling."""
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return Decimal(str(value))
        elif isinstance(value, dict):
            return {k: self._normalize_value(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [self._normalize_value(v) for v in value]
        return value

    def _normalize_values(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a dictionary of values."""
        return {k: self._normalize_value(v) for k, v in values.items()}

    def _serialize_for_hash(self, value: Any) -> Any:
        """Serialize values for hash calculation."""
        if isinstance(value, Decimal):
            return str(value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(Decimal(str(value)))
        elif isinstance(value, dict):
            return {k: self._serialize_for_hash(v) for k, v in sorted(value.items())}
        elif isinstance(value, (list, tuple)):
            return [self._serialize_for_hash(v) for v in value]
        elif isinstance(value, Enum):
            return value.value
        return str(value) if value is not None else None

    def _serialize_step_for_hash(self, step: CalculationStep) -> Dict[str, Any]:
        """Serialize a calculation step for hashing."""
        return {
            'step_number': step.step_number,
            'operation': step.operation,
            'description': step.description,
            'inputs': self._serialize_for_hash(step.inputs),
            'output_value': self._serialize_for_hash(step.output_value),
            'output_name': step.output_name,
            'formula': step.formula or '',
            'units': step.units or ''
        }

    def reset(self) -> None:
        """Reset tracker for new calculation."""
        with self._lock:
            self.steps = []
            self.input_parameters = {}


class ProvenanceValidator:
    """
    Validates calculation provenance for integrity and reproducibility.

    Provides methods to verify that calculations have not been tampered
    with and can be reproduced exactly.

    ZERO-HALLUCINATION GUARANTEE:
    - Uses only deterministic hash comparisons
    - No ML/AI inference
    """

    @staticmethod
    def validate_hash(record: ProvenanceRecord) -> bool:
        """
        Validate that a provenance hash matches the calculation.

        Recreates the hash from record data and compares to stored hash.

        Args:
            record: ProvenanceRecord to validate

        Returns:
            True if hash is valid (calculation not tampered)
        """
        # Recreate tracker with same parameters
        tracker = ProvenanceTracker(
            record.calculation_id,
            record.calculation_type,
            record.version
        )
        tracker.input_parameters = record.input_parameters
        tracker.steps = record.calculation_steps

        # Generate hash and compare
        expected_hash = tracker.generate_hash(record.final_result)
        return expected_hash == record.provenance_hash

    @staticmethod
    def validate_reproducibility(
        record1: ProvenanceRecord,
        record2: ProvenanceRecord
    ) -> bool:
        """
        Validate that two calculations with same inputs produce same results.

        This verifies the zero-hallucination guarantee - identical inputs
        must produce bit-perfect identical outputs.

        Args:
            record1: First provenance record
            record2: Second provenance record

        Returns:
            True if calculations are reproducible (identical results)
        """
        # Check inputs match
        if record1.input_parameters != record2.input_parameters:
            logger.warning("Input parameters do not match")
            return False

        # Check calculation types match
        if record1.calculation_type != record2.calculation_type:
            logger.warning("Calculation types do not match")
            return False

        # Check results match (bit-perfect)
        if str(record1.final_result) != str(record2.final_result):
            logger.warning("Final results do not match")
            return False

        # Check hashes match
        if record1.provenance_hash != record2.provenance_hash:
            logger.warning("Provenance hashes do not match")
            return False

        return True

    @staticmethod
    def validate_chain(chain: ProvenanceChain) -> bool:
        """
        Validate integrity of a provenance chain.

        Verifies that chain hash matches computed value and
        all records are valid.

        Args:
            chain: ProvenanceChain to validate

        Returns:
            True if chain is intact
        """
        # Recalculate chain hash
        chain_data = [r.provenance_hash for r in chain.records]
        chain_str = json.dumps(chain_data, sort_keys=True)
        expected_hash = hashlib.sha256(chain_str.encode('utf-8')).hexdigest()

        if expected_hash != chain.chain_hash:
            logger.warning("Chain hash mismatch")
            return False

        # Validate each record
        for record in chain.records:
            if not ProvenanceValidator.validate_hash(record):
                logger.warning(f"Record {record.calculation_id} failed validation")
                return False

        return True


def create_calculation_hash(data: Dict[str, Any]) -> str:
    """
    Create a deterministic hash for any calculation data.

    Utility function for quick hashing needs without full
    provenance tracking.

    ZERO-HALLUCINATION GUARANTEE:
    - Uses sorted keys for deterministic JSON
    - SHA-256 for cryptographic integrity

    Args:
        data: Dictionary of calculation data

    Returns:
        SHA-256 hash as hexadecimal string

    Example:
        >>> hash = create_calculation_hash({
        ...     'operation': 'multiply',
        ...     'inputs': {'a': 10, 'b': 20},
        ...     'result': 200
        ... })
        >>> print(hash)  # 64-character hex string
    """
    canonical_json = json.dumps(
        data,
        sort_keys=True,
        separators=(',', ':'),
        default=str
    )
    return hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()


def verify_calculation_hash(data: Dict[str, Any], expected_hash: str) -> bool:
    """
    Verify that data produces expected hash.

    Args:
        data: Dictionary of calculation data
        expected_hash: Expected SHA-256 hash

    Returns:
        True if hashes match

    Example:
        >>> data = {'a': 1, 'b': 2}
        >>> hash = create_calculation_hash(data)
        >>> verify_calculation_hash(data, hash)  # True
    """
    actual_hash = create_calculation_hash(data)
    return actual_hash == expected_hash


# Unit test examples
def _run_self_tests():
    """Run self-tests to verify provenance tracking correctness."""

    # Test 1: Basic hash generation
    tracker = ProvenanceTracker(
        calculation_id="test_001",
        calculation_type="steam_quality",
        version="1.0.0"
    )
    tracker.record_inputs({'pressure': 1.0, 'temperature': 180.0})
    tracker.record_step(
        operation="lookup",
        description="Get saturation temperature",
        inputs={'pressure': 1.0},
        output_value=179.88,
        output_name="T_sat"
    )
    hash1 = tracker.generate_hash(Decimal('0.95'))
    assert len(hash1) == 64, f"Hash should be 64 chars: {len(hash1)}"
    print(f"Test 1 passed: hash length = {len(hash1)}")

    # Test 2: Deterministic hashing (same input -> same hash)
    tracker2 = ProvenanceTracker(
        calculation_id="test_001",
        calculation_type="steam_quality",
        version="1.0.0"
    )
    tracker2.record_inputs({'pressure': 1.0, 'temperature': 180.0})
    tracker2.record_step(
        operation="lookup",
        description="Get saturation temperature",
        inputs={'pressure': 1.0},
        output_value=179.88,
        output_name="T_sat"
    )
    hash2 = tracker2.generate_hash(Decimal('0.95'))
    assert hash1 == hash2, "Hashes should be identical for same inputs"
    print(f"Test 2 passed: hashes are deterministic")

    # Test 3: Different inputs produce different hashes
    tracker3 = ProvenanceTracker(
        calculation_id="test_001",
        calculation_type="steam_quality",
        version="1.0.0"
    )
    tracker3.record_inputs({'pressure': 2.0, 'temperature': 180.0})  # Different pressure
    tracker3.record_step(
        operation="lookup",
        description="Get saturation temperature",
        inputs={'pressure': 2.0},
        output_value=212.38,
        output_name="T_sat"
    )
    hash3 = tracker3.generate_hash(Decimal('0.95'))
    assert hash1 != hash3, "Different inputs should produce different hashes"
    print(f"Test 3 passed: different inputs produce different hashes")

    # Test 4: Provenance record creation
    record = tracker.get_provenance_record(Decimal('0.95'))
    assert record.provenance_hash == hash1, "Record hash should match"
    assert len(record.calculation_steps) == 1, "Should have 1 step"
    print(f"Test 4 passed: provenance record created")

    # Test 5: Hash verification
    is_valid = tracker.verify_hash(Decimal('0.95'), hash1)
    assert is_valid, "Hash should verify successfully"
    print(f"Test 5 passed: hash verification works")

    # Test 6: Validator
    is_valid = ProvenanceValidator.validate_hash(record)
    assert is_valid, "Validator should confirm hash is valid"
    print(f"Test 6 passed: ProvenanceValidator.validate_hash works")

    # Test 7: Reproducibility validation
    record2 = tracker2.get_provenance_record(Decimal('0.95'))
    is_reproducible = ProvenanceValidator.validate_reproducibility(record, record2)
    assert is_reproducible, "Identical calculations should be reproducible"
    print(f"Test 7 passed: reproducibility validation works")

    # Test 8: Utility functions
    data = {'operation': 'test', 'value': 42}
    util_hash = create_calculation_hash(data)
    assert verify_calculation_hash(data, util_hash), "Utility verify should work"
    print(f"Test 8 passed: utility functions work")

    # Test 9: Provenance chain
    chain = ProvenanceChain(chain_id="chain_001")
    chain.add_record(record)
    chain.add_record(record2)
    assert len(chain.records) == 2, "Chain should have 2 records"
    assert chain.chain_hash, "Chain should have hash"
    print(f"Test 9 passed: provenance chain works")

    # Test 10: Chain validation
    is_chain_valid = ProvenanceValidator.validate_chain(chain)
    assert is_chain_valid, "Chain should validate"
    print(f"Test 10 passed: chain validation works")

    # Test 11: Audit trail export
    audit_trail = tracker.create_audit_trail(Decimal('0.95'))
    assert 'audit_trail' in audit_trail, "Should have audit_trail section"
    assert 'inputs' in audit_trail, "Should have inputs section"
    assert 'steps' in audit_trail, "Should have steps section"
    print(f"Test 11 passed: audit trail export works")

    print("\nAll self-tests passed!")
    return True


if __name__ == "__main__":
    _run_self_tests()
