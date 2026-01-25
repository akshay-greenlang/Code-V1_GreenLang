"""
GL-019 HEATSCHEDULER - Calculation Provenance Module

This module provides SHA-256 hashing and audit trail capabilities for
zero-hallucination, deterministic calculations in heating schedule optimization.

All calculations are fully traceable with cryptographic verification,
ensuring 100% reproducibility and regulatory compliance.

Standards Reference:
- GreenLang Zero-Hallucination Guarantee
- ISO 50001 - Energy Management Systems
- ISO 50006 - Measuring Energy Performance Using Baselines
- ASHRAE Guideline 14 - Measurement of Energy and Demand Savings

Author: GL-CalculatorEngineer
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Union
import hashlib
import json


@dataclass(frozen=True)
class CalculationStep:
    """
    Immutable record of a single calculation step.

    Attributes:
        step_number: Sequential step identifier
        description: Human-readable description of the step
        operation: Mathematical operation performed
        inputs: Dictionary of input values used
        output_value: Result of the calculation
        output_name: Name/identifier of the output
        formula: Mathematical formula applied (LaTeX or text)
    """
    step_number: int
    description: str
    operation: str
    inputs: Dict[str, Any]
    output_value: Union[float, Decimal, str, List, Dict]
    output_name: str
    formula: str = ""


@dataclass(frozen=True)
class ProvenanceRecord:
    """
    Complete provenance record for a calculation.

    This record provides a cryptographically verifiable audit trail
    for regulatory compliance and zero-hallucination guarantee.

    Attributes:
        calculation_id: Unique identifier for this calculation
        calculator_name: Name of the calculator module used
        calculator_version: Version of the calculator
        timestamp_utc: ISO 8601 timestamp in UTC
        input_hash: SHA-256 hash of all inputs
        output_hash: SHA-256 hash of all outputs
        provenance_hash: SHA-256 hash of complete calculation
        inputs: Original input parameters
        outputs: Calculated output values
        calculation_steps: List of all calculation steps
        metadata: Additional metadata (standards, references)
    """
    calculation_id: str
    calculator_name: str
    calculator_version: str
    timestamp_utc: str
    input_hash: str
    output_hash: str
    provenance_hash: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    calculation_steps: List[CalculationStep]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProvenanceTracker:
    """
    Tracks calculation provenance with SHA-256 hashing for audit trail.

    This class ensures all calculations are:
    - Deterministic: Same input always produces same output
    - Reproducible: Complete audit trail with step-by-step tracking
    - Verifiable: SHA-256 hashes for cryptographic verification
    - Auditable: Full provenance chain for regulatory compliance

    Example:
        >>> tracker = ProvenanceTracker("EnergyCostCalculator", "1.0.0")
        >>> tracker.set_inputs({"energy_kwh": 1000.0, "rate_per_kwh": 0.12})
        >>> tracker.add_step(1, "Calculate base cost", "multiply",
        ...                  {"energy_kwh": 1000.0, "rate": 0.12}, 120.0, "base_cost")
        >>> tracker.set_outputs({"total_cost_usd": 120.0})
        >>> record = tracker.finalize()
        >>> print(record.provenance_hash)
    """

    def __init__(
        self,
        calculator_name: str,
        calculator_version: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize provenance tracker.

        Args:
            calculator_name: Name of the calculator module
            calculator_version: Semantic version of the calculator
            metadata: Optional metadata (standards, references, etc.)
        """
        self._calculator_name = calculator_name
        self._calculator_version = calculator_version
        self._metadata = metadata or {}
        self._inputs: Dict[str, Any] = {}
        self._outputs: Dict[str, Any] = {}
        self._steps: List[CalculationStep] = []
        self._timestamp: Optional[str] = None
        self._calculation_id: Optional[str] = None

    def set_inputs(self, inputs: Dict[str, Any]) -> None:
        """
        Set input parameters for the calculation.

        Args:
            inputs: Dictionary of input parameter names and values
        """
        self._inputs = self._normalize_values(inputs)
        self._timestamp = datetime.now(timezone.utc).isoformat()
        self._calculation_id = self._generate_calculation_id()

    def add_step(
        self,
        step_number: int,
        description: str,
        operation: str,
        inputs: Dict[str, Any],
        output_value: Union[float, Decimal, str, List, Dict],
        output_name: str,
        formula: str = ""
    ) -> None:
        """
        Add a calculation step to the provenance trail.

        Args:
            step_number: Sequential step identifier
            description: Human-readable description
            operation: Mathematical operation (add, multiply, etc.)
            inputs: Input values for this step
            output_value: Result of this step
            output_name: Name of the output variable
            formula: Mathematical formula (optional, for documentation)
        """
        step = CalculationStep(
            step_number=step_number,
            description=description,
            operation=operation,
            inputs=self._normalize_values(inputs),
            output_value=self._normalize_single_value(output_value),
            output_name=output_name,
            formula=formula
        )
        self._steps.append(step)

    def set_outputs(self, outputs: Dict[str, Any]) -> None:
        """
        Set output values from the calculation.

        Args:
            outputs: Dictionary of output names and values
        """
        self._outputs = self._normalize_values(outputs)

    def finalize(self) -> ProvenanceRecord:
        """
        Finalize the provenance record with SHA-256 hashes.

        Returns:
            Complete ProvenanceRecord with cryptographic verification

        Raises:
            ValueError: If inputs or outputs not set
        """
        if not self._inputs:
            raise ValueError("Inputs must be set before finalizing")
        if not self._outputs:
            raise ValueError("Outputs must be set before finalizing")
        if self._timestamp is None:
            raise ValueError("Timestamp not set - call set_inputs first")
        if self._calculation_id is None:
            raise ValueError("Calculation ID not generated")

        input_hash = self._compute_hash(self._inputs)
        output_hash = self._compute_hash(self._outputs)

        # Compute provenance hash from all components
        provenance_data = {
            "calculation_id": self._calculation_id,
            "calculator_name": self._calculator_name,
            "calculator_version": self._calculator_version,
            "timestamp_utc": self._timestamp,
            "input_hash": input_hash,
            "output_hash": output_hash,
            "inputs": self._inputs,
            "outputs": self._outputs,
            "steps": [self._step_to_dict(s) for s in self._steps],
            "metadata": self._metadata
        }
        provenance_hash = self._compute_hash(provenance_data)

        return ProvenanceRecord(
            calculation_id=self._calculation_id,
            calculator_name=self._calculator_name,
            calculator_version=self._calculator_version,
            timestamp_utc=self._timestamp,
            input_hash=input_hash,
            output_hash=output_hash,
            provenance_hash=provenance_hash,
            inputs=self._inputs,
            outputs=self._outputs,
            calculation_steps=self._steps,
            metadata=self._metadata
        )

    def _generate_calculation_id(self) -> str:
        """Generate unique calculation ID based on inputs and timestamp."""
        id_data = {
            "calculator": self._calculator_name,
            "version": self._calculator_version,
            "timestamp": self._timestamp,
            "inputs": self._inputs
        }
        full_hash = self._compute_hash(id_data)
        # Return first 16 characters for readability
        return f"CALC-{full_hash[:16].upper()}"

    def _compute_hash(self, data: Any) -> str:
        """
        Compute SHA-256 hash of data.

        Uses deterministic JSON serialization for reproducibility.

        Args:
            data: Data to hash (must be JSON-serializable)

        Returns:
            Hexadecimal SHA-256 hash string
        """
        # Sort keys for deterministic serialization
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()

    def _normalize_values(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize all values in dictionary for consistent hashing."""
        return {k: self._normalize_single_value(v) for k, v in data.items()}

    def _normalize_single_value(self, value: Any) -> Any:
        """
        Normalize a single value for consistent hashing.

        Converts floats to strings with consistent precision,
        handles Decimal types, and ensures deterministic serialization.
        """
        if isinstance(value, Decimal):
            return str(value)
        elif isinstance(value, float):
            # Use consistent precision for floats
            return round(value, 10)
        elif isinstance(value, dict):
            return self._normalize_values(value)
        elif isinstance(value, (list, tuple)):
            return [self._normalize_single_value(v) for v in value]
        return value

    def _step_to_dict(self, step: CalculationStep) -> Dict[str, Any]:
        """Convert CalculationStep to dictionary for hashing."""
        return {
            "step_number": step.step_number,
            "description": step.description,
            "operation": step.operation,
            "inputs": step.inputs,
            "output_value": self._normalize_single_value(step.output_value),
            "output_name": step.output_name,
            "formula": step.formula
        }


def compute_input_fingerprint(inputs: Dict[str, Any]) -> str:
    """
    Compute SHA-256 fingerprint of calculation inputs.

    This function provides a quick way to verify input integrity
    without creating a full provenance record.

    Args:
        inputs: Dictionary of input parameters

    Returns:
        SHA-256 hash of inputs (first 16 chars for readability)

    Example:
        >>> fingerprint = compute_input_fingerprint({"energy_kwh": 1000.0})
        >>> print(fingerprint)  # e.g., "a1b2c3d4e5f6g7h8"
    """
    json_str = json.dumps(inputs, sort_keys=True, default=str)
    full_hash = hashlib.sha256(json_str.encode('utf-8')).hexdigest()
    return full_hash[:16]


def compute_output_fingerprint(outputs: Dict[str, Any]) -> str:
    """
    Compute SHA-256 fingerprint of calculation outputs.

    Args:
        outputs: Dictionary of output values

    Returns:
        SHA-256 hash of outputs (first 16 chars)
    """
    json_str = json.dumps(outputs, sort_keys=True, default=str)
    full_hash = hashlib.sha256(json_str.encode('utf-8')).hexdigest()
    return full_hash[:16]


def verify_provenance(record: ProvenanceRecord) -> bool:
    """
    Verify the integrity of a provenance record.

    Recomputes all hashes and compares with stored values
    to detect any tampering or corruption.

    Args:
        record: ProvenanceRecord to verify

    Returns:
        True if all hashes match, False otherwise

    Example:
        >>> record = tracker.finalize()
        >>> is_valid = verify_provenance(record)
        >>> assert is_valid, "Provenance verification failed!"
    """
    # Recompute input hash
    input_json = json.dumps(record.inputs, sort_keys=True, default=str)
    computed_input_hash = hashlib.sha256(input_json.encode('utf-8')).hexdigest()

    if computed_input_hash != record.input_hash:
        return False

    # Recompute output hash
    output_json = json.dumps(record.outputs, sort_keys=True, default=str)
    computed_output_hash = hashlib.sha256(output_json.encode('utf-8')).hexdigest()

    if computed_output_hash != record.output_hash:
        return False

    # Recompute provenance hash
    provenance_data = {
        "calculation_id": record.calculation_id,
        "calculator_name": record.calculator_name,
        "calculator_version": record.calculator_version,
        "timestamp_utc": record.timestamp_utc,
        "input_hash": record.input_hash,
        "output_hash": record.output_hash,
        "inputs": record.inputs,
        "outputs": record.outputs,
        "steps": [
            {
                "step_number": s.step_number,
                "description": s.description,
                "operation": s.operation,
                "inputs": s.inputs,
                "output_value": s.output_value if not isinstance(s.output_value, Decimal)
                               else str(s.output_value),
                "output_name": s.output_name,
                "formula": s.formula
            }
            for s in record.calculation_steps
        ],
        "metadata": record.metadata
    }

    provenance_json = json.dumps(provenance_data, sort_keys=True, default=str)
    computed_provenance_hash = hashlib.sha256(provenance_json.encode('utf-8')).hexdigest()

    return computed_provenance_hash == record.provenance_hash


def get_utc_timestamp() -> str:
    """
    Get current UTC timestamp in ISO 8601 format.

    Returns:
        ISO 8601 formatted UTC timestamp string
    """
    return datetime.now(timezone.utc).isoformat()


def format_provenance_report(record: ProvenanceRecord) -> str:
    """
    Format a provenance record as a human-readable report.

    Args:
        record: ProvenanceRecord to format

    Returns:
        Formatted string report suitable for audit logs
    """
    lines = [
        "=" * 70,
        "CALCULATION PROVENANCE REPORT - GL-019 HEATSCHEDULER",
        "=" * 70,
        f"Calculation ID: {record.calculation_id}",
        f"Calculator: {record.calculator_name} v{record.calculator_version}",
        f"Timestamp (UTC): {record.timestamp_utc}",
        "-" * 70,
        "CRYPTOGRAPHIC VERIFICATION",
        f"  Input Hash:      {record.input_hash}",
        f"  Output Hash:     {record.output_hash}",
        f"  Provenance Hash: {record.provenance_hash}",
        "-" * 70,
        "INPUTS",
    ]

    for key, value in record.inputs.items():
        lines.append(f"  {key}: {value}")

    lines.append("-" * 70)
    lines.append("CALCULATION STEPS")

    for step in record.calculation_steps:
        lines.append(f"  Step {step.step_number}: {step.description}")
        lines.append(f"    Operation: {step.operation}")
        if step.formula:
            lines.append(f"    Formula: {step.formula}")
        lines.append(f"    Inputs: {step.inputs}")
        lines.append(f"    Output: {step.output_name} = {step.output_value}")

    lines.append("-" * 70)
    lines.append("OUTPUTS")

    for key, value in record.outputs.items():
        lines.append(f"  {key}: {value}")

    if record.metadata:
        lines.append("-" * 70)
        lines.append("METADATA")
        for key, value in record.metadata.items():
            lines.append(f"  {key}: {value}")

    lines.append("=" * 70)

    return "\n".join(lines)
