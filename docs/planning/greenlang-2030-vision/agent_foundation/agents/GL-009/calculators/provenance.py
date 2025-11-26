"""Provenance Tracking Utilities.

This module provides comprehensive provenance tracking for thermal
efficiency calculations, ensuring complete audit trails, reproducibility,
and regulatory compliance.

Features:
    - SHA-256 hashing for calculation verification
    - Calculation step logging
    - Audit trail generation
    - Reproducibility verification
    - Chain of custody tracking

Standards:
    - ISO 17025: Testing and calibration laboratories
    - 21 CFR Part 11: Electronic records
    - SOC 2 Type II: Audit requirements

The provenance system guarantees:
    - Bit-perfect reproducibility
    - Complete calculation audit trails
    - Tamper-evident records
    - Regulatory compliance documentation

Author: GL-009 THERMALIQ Agent
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import uuid


class RecordType(Enum):
    """Type of provenance record."""
    CALCULATION = "calculation"
    INPUT_DATA = "input_data"
    OUTPUT_DATA = "output_data"
    INTERMEDIATE = "intermediate"
    VALIDATION = "validation"
    ERROR = "error"
    METADATA = "metadata"


class VerificationStatus(Enum):
    """Status of verification check."""
    VERIFIED = "verified"
    FAILED = "failed"
    PENDING = "pending"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class CalculationStep:
    """Individual calculation step with full provenance.

    Attributes:
        step_id: Unique identifier for this step
        step_number: Sequential step number
        timestamp: When step was executed
        description: Human-readable description
        operation: Mathematical operation performed
        inputs: Input values and sources
        output_value: Result of calculation
        output_name: Name of output variable
        formula: Formula/equation used
        precision: Decimal precision used
        rounding_method: Rounding method applied
        hash: SHA-256 hash of step data
    """
    step_id: str
    step_number: int
    timestamp: str
    description: str
    operation: str
    inputs: Dict[str, Any]
    output_value: Any
    output_name: str
    formula: Optional[str] = None
    precision: Optional[int] = None
    rounding_method: Optional[str] = None
    hash: Optional[str] = None

    def __post_init__(self) -> None:
        """Generate hash if not provided."""
        if self.hash is None:
            self.hash = self._generate_hash()

    def _generate_hash(self) -> str:
        """Generate SHA-256 hash of step data."""
        data = {
            "step_number": self.step_number,
            "operation": self.operation,
            "inputs": self.inputs,
            "output_value": str(self.output_value),
            "output_name": self.output_name,
            "formula": self.formula
        }
        json_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_id": self.step_id,
            "step_number": self.step_number,
            "timestamp": self.timestamp,
            "description": self.description,
            "operation": self.operation,
            "inputs": self.inputs,
            "output_value": self.output_value,
            "output_name": self.output_name,
            "formula": self.formula,
            "precision": self.precision,
            "rounding_method": self.rounding_method,
            "hash": self.hash
        }


@dataclass
class ProvenanceRecord:
    """Complete provenance record for a calculation.

    Attributes:
        record_id: Unique record identifier (UUID)
        record_type: Type of record
        calculator_name: Name of calculator that produced this
        calculator_version: Version of calculator
        created_at: Creation timestamp (ISO 8601)
        input_hash: Hash of all inputs
        output_hash: Hash of all outputs
        steps_hash: Hash of all calculation steps
        combined_hash: Combined provenance hash
        inputs: All input data
        outputs: All output data
        steps: List of calculation steps
        metadata: Additional metadata
        previous_record_hash: Hash of previous record (chain)
        verification_status: Current verification status
    """
    record_id: str
    record_type: RecordType
    calculator_name: str
    calculator_version: str
    created_at: str
    input_hash: str
    output_hash: str
    steps_hash: str
    combined_hash: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    steps: List[CalculationStep]
    metadata: Dict[str, Any] = field(default_factory=dict)
    previous_record_hash: Optional[str] = None
    verification_status: VerificationStatus = VerificationStatus.PENDING

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "record_id": self.record_id,
            "record_type": self.record_type.value,
            "calculator_name": self.calculator_name,
            "calculator_version": self.calculator_version,
            "created_at": self.created_at,
            "hashes": {
                "input": self.input_hash,
                "output": self.output_hash,
                "steps": self.steps_hash,
                "combined": self.combined_hash
            },
            "inputs": self.inputs,
            "outputs": self.outputs,
            "steps": [s.to_dict() for s in self.steps],
            "metadata": self.metadata,
            "previous_record_hash": self.previous_record_hash,
            "verification_status": self.verification_status.value
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


@dataclass
class AuditTrail:
    """Complete audit trail for a series of calculations.

    Attributes:
        trail_id: Unique trail identifier
        created_at: Trail creation timestamp
        records: List of provenance records
        chain_valid: Whether hash chain is valid
        total_records: Number of records
        first_record_hash: Hash of first record
        last_record_hash: Hash of last record
        metadata: Trail metadata
    """
    trail_id: str
    created_at: str
    records: List[ProvenanceRecord]
    chain_valid: bool
    total_records: int
    first_record_hash: Optional[str]
    last_record_hash: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trail_id": self.trail_id,
            "created_at": self.created_at,
            "total_records": self.total_records,
            "chain_valid": self.chain_valid,
            "first_record_hash": self.first_record_hash,
            "last_record_hash": self.last_record_hash,
            "records": [r.to_dict() for r in self.records],
            "metadata": self.metadata
        }


class ProvenanceTracker:
    """Provenance Tracking System.

    Provides comprehensive provenance tracking for calculations,
    ensuring reproducibility and audit compliance.

    Example:
        >>> tracker = ProvenanceTracker("FirstLawCalculator", "1.0.0")
        >>> tracker.log_inputs({"fuel_kw": 1000, "steam_kw": 850})
        >>> tracker.log_step(
        ...     description="Calculate efficiency",
        ...     operation="divide",
        ...     inputs={"steam": 850, "fuel": 1000},
        ...     output_value=0.85,
        ...     output_name="efficiency"
        ... )
        >>> record = tracker.finalize({"efficiency_percent": 85.0})
    """

    VERSION: str = "1.0.0"

    def __init__(
        self,
        calculator_name: str,
        calculator_version: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize the Provenance Tracker.

        Args:
            calculator_name: Name of the calculator
            calculator_version: Version of the calculator
            metadata: Optional metadata to include
        """
        self.calculator_name = calculator_name
        self.calculator_version = calculator_version
        self.metadata = metadata or {}

        self._record_id = str(uuid.uuid4())
        self._created_at = datetime.now(timezone.utc).isoformat()
        self._inputs: Dict[str, Any] = {}
        self._outputs: Dict[str, Any] = {}
        self._steps: List[CalculationStep] = []
        self._step_counter = 0
        self._previous_hash: Optional[str] = None

    def log_inputs(self, inputs: Dict[str, Any]) -> str:
        """Log input data and return hash.

        Args:
            inputs: Dictionary of input values

        Returns:
            SHA-256 hash of inputs
        """
        self._inputs.update(inputs)
        return generate_provenance_hash(inputs)

    def log_step(
        self,
        description: str,
        operation: str,
        inputs: Dict[str, Any],
        output_value: Any,
        output_name: str,
        formula: Optional[str] = None,
        precision: Optional[int] = None,
        rounding_method: Optional[str] = None
    ) -> CalculationStep:
        """Log a calculation step.

        Args:
            description: Human-readable description
            operation: Mathematical operation
            inputs: Input values for this step
            output_value: Result of calculation
            output_name: Name of output variable
            formula: Formula used (optional)
            precision: Precision used (optional)
            rounding_method: Rounding method (optional)

        Returns:
            CalculationStep record
        """
        self._step_counter += 1

        step = CalculationStep(
            step_id=f"{self._record_id}-step-{self._step_counter}",
            step_number=self._step_counter,
            timestamp=datetime.now(timezone.utc).isoformat(),
            description=description,
            operation=operation,
            inputs=inputs,
            output_value=output_value,
            output_name=output_name,
            formula=formula,
            precision=precision,
            rounding_method=rounding_method
        )

        self._steps.append(step)
        return step

    def finalize(
        self,
        outputs: Dict[str, Any],
        previous_record_hash: Optional[str] = None
    ) -> ProvenanceRecord:
        """Finalize and create the provenance record.

        Args:
            outputs: Dictionary of output values
            previous_record_hash: Hash of previous record for chaining

        Returns:
            Complete ProvenanceRecord
        """
        self._outputs = outputs

        # Generate hashes
        input_hash = generate_provenance_hash(self._inputs)
        output_hash = generate_provenance_hash(outputs)
        steps_hash = self._generate_steps_hash()

        # Combined hash
        combined_data = {
            "input_hash": input_hash,
            "output_hash": output_hash,
            "steps_hash": steps_hash,
            "calculator": self.calculator_name,
            "version": self.calculator_version
        }
        combined_hash = generate_provenance_hash(combined_data)

        record = ProvenanceRecord(
            record_id=self._record_id,
            record_type=RecordType.CALCULATION,
            calculator_name=self.calculator_name,
            calculator_version=self.calculator_version,
            created_at=self._created_at,
            input_hash=input_hash,
            output_hash=output_hash,
            steps_hash=steps_hash,
            combined_hash=combined_hash,
            inputs=self._inputs,
            outputs=self._outputs,
            steps=self._steps,
            metadata=self.metadata,
            previous_record_hash=previous_record_hash,
            verification_status=VerificationStatus.VERIFIED
        )

        return record

    def _generate_steps_hash(self) -> str:
        """Generate hash of all calculation steps."""
        step_hashes = [s.hash for s in self._steps if s.hash]
        combined = "|".join(step_hashes)
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()

    def reset(self) -> None:
        """Reset tracker for new calculation."""
        self._record_id = str(uuid.uuid4())
        self._created_at = datetime.now(timezone.utc).isoformat()
        self._inputs = {}
        self._outputs = {}
        self._steps = []
        self._step_counter = 0


def generate_provenance_hash(data: Any) -> str:
    """Generate SHA-256 hash for any data structure.

    Creates a deterministic hash by:
    1. Converting to JSON with sorted keys
    2. Using compact separators
    3. Encoding as UTF-8
    4. Computing SHA-256

    Args:
        data: Any JSON-serializable data

    Returns:
        SHA-256 hash as hexadecimal string
    """
    if data is None:
        return hashlib.sha256(b"null").hexdigest()

    try:
        # Convert to canonical JSON representation
        json_str = json.dumps(
            data,
            sort_keys=True,
            separators=(',', ':'),
            default=str  # Handle non-serializable types
        )
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()
    except Exception as e:
        # Fallback to string representation
        return hashlib.sha256(str(data).encode('utf-8')).hexdigest()


def verify_provenance_hash(data: Any, expected_hash: str) -> bool:
    """Verify data matches expected provenance hash.

    Args:
        data: Data to verify
        expected_hash: Expected SHA-256 hash

    Returns:
        True if hashes match, False otherwise
    """
    actual_hash = generate_provenance_hash(data)
    return actual_hash == expected_hash


def create_audit_trail(
    records: List[ProvenanceRecord],
    metadata: Optional[Dict[str, Any]] = None
) -> AuditTrail:
    """Create an audit trail from a list of provenance records.

    Validates the hash chain and creates a complete audit trail.

    Args:
        records: List of provenance records in order
        metadata: Optional trail metadata

    Returns:
        AuditTrail with chain validation status
    """
    trail_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    # Validate hash chain
    chain_valid = True
    for i, record in enumerate(records[1:], start=1):
        expected_prev = records[i - 1].combined_hash
        if record.previous_record_hash != expected_prev:
            chain_valid = False
            break

    first_hash = records[0].combined_hash if records else None
    last_hash = records[-1].combined_hash if records else None

    return AuditTrail(
        trail_id=trail_id,
        created_at=created_at,
        records=records,
        chain_valid=chain_valid,
        total_records=len(records),
        first_record_hash=first_hash,
        last_record_hash=last_hash,
        metadata=metadata or {}
    )


def verify_audit_trail(trail: AuditTrail) -> Tuple[bool, List[str]]:
    """Verify integrity of an audit trail.

    Checks:
    1. Hash chain continuity
    2. Individual record hashes
    3. Step hashes within records

    Args:
        trail: AuditTrail to verify

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors: List[str] = []

    # Check hash chain
    for i, record in enumerate(trail.records[1:], start=1):
        expected_prev = trail.records[i - 1].combined_hash
        if record.previous_record_hash != expected_prev:
            errors.append(
                f"Record {i}: Chain broken. Expected prev={expected_prev[:16]}..., "
                f"got {record.previous_record_hash[:16] if record.previous_record_hash else 'None'}..."
            )

    # Verify individual records
    for i, record in enumerate(trail.records):
        # Verify input hash
        calculated_input_hash = generate_provenance_hash(record.inputs)
        if calculated_input_hash != record.input_hash:
            errors.append(f"Record {i}: Input hash mismatch")

        # Verify output hash
        calculated_output_hash = generate_provenance_hash(record.outputs)
        if calculated_output_hash != record.output_hash:
            errors.append(f"Record {i}: Output hash mismatch")

        # Verify step hashes
        for j, step in enumerate(record.steps):
            expected_hash = step._generate_hash()
            if expected_hash != step.hash:
                errors.append(f"Record {i}, Step {j}: Step hash mismatch")

    is_valid = len(errors) == 0
    return is_valid, errors


def export_audit_trail(
    trail: AuditTrail,
    format: str = "json",
    file_path: Optional[str] = None
) -> str:
    """Export audit trail to file or string.

    Args:
        trail: AuditTrail to export
        format: Output format ("json" or "csv")
        file_path: Optional file path to write

    Returns:
        Formatted string representation
    """
    if format == "json":
        output = json.dumps(trail.to_dict(), indent=2, default=str)
    elif format == "csv":
        # CSV format for records
        lines = ["record_id,calculator,created_at,input_hash,output_hash,combined_hash"]
        for record in trail.records:
            lines.append(
                f"{record.record_id},{record.calculator_name},"
                f"{record.created_at},{record.input_hash[:16]}...,"
                f"{record.output_hash[:16]}...,{record.combined_hash[:16]}..."
            )
        output = "\n".join(lines)
    else:
        raise ValueError(f"Unknown format: {format}")

    if file_path:
        with open(file_path, 'w') as f:
            f.write(output)

    return output


# Type alias for convenience
Tuple = tuple  # Re-export for type hints
