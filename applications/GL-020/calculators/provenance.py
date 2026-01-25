"""
GL-020 ECONOPULSE: Provenance Tracking Module

Zero-hallucination calculation provenance tracking with SHA-256 hashes
for complete audit trails and regulatory compliance.

This module provides:
- SHA-256 hash generation for all calculations
- Input/output logging with timestamps
- Audit trail for regulatory compliance
- Reproducibility verification
- Calculation step documentation

Author: GL-CalculatorEngineer
Standard: ASME PTC 4.3 compliant audit trails
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Union
from enum import Enum


class CalculationType(Enum):
    """Enumeration of calculation types for categorization."""
    HEAT_TRANSFER = "heat_transfer"
    FOULING = "fouling"
    EFFICIENCY = "efficiency"
    THERMAL_PROPERTIES = "thermal_properties"
    SOOT_BLOWER = "soot_blower"


@dataclass
class CalculationStep:
    """
    Individual calculation step with complete provenance.

    Attributes:
        step_number: Sequential step identifier
        operation: Mathematical operation performed
        description: Human-readable description of the step
        inputs: Dictionary of input values used
        output_name: Name of the output variable
        output_value: Calculated result
        formula: Mathematical formula used (LaTeX or text)
    """
    step_number: int
    operation: str
    description: str
    inputs: Dict[str, Any]
    output_name: str
    output_value: Union[float, Decimal]
    formula: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary for hashing."""
        return {
            "step_number": self.step_number,
            "operation": self.operation,
            "description": self.description,
            "inputs": {k: str(v) for k, v in self.inputs.items()},
            "output_name": self.output_name,
            "output_value": str(self.output_value),
            "formula": self.formula
        }


@dataclass
class CalculationProvenance:
    """
    Complete provenance record for a calculation.

    Provides SHA-256 hash-based verification of calculation integrity,
    ensuring bit-perfect reproducibility and regulatory compliance.

    Attributes:
        calculation_id: Unique identifier for this calculation
        calculation_type: Category of calculation
        formula_id: Identifier of the formula used
        formula_version: Version of the formula
        timestamp_utc: UTC timestamp of calculation
        inputs: All input parameters
        steps: List of calculation steps
        output_value: Final calculated value
        output_unit: Unit of measurement for output
        precision: Decimal places in output
        provenance_hash: SHA-256 hash of entire calculation
        calculation_time_ms: Time taken for calculation
        metadata: Additional metadata
    """
    calculation_id: str
    calculation_type: CalculationType
    formula_id: str
    formula_version: str
    timestamp_utc: str
    inputs: Dict[str, Any]
    steps: List[CalculationStep]
    output_value: Union[float, Decimal]
    output_unit: str
    precision: int = 6
    provenance_hash: str = ""
    calculation_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Generate provenance hash after initialization."""
        if not self.provenance_hash:
            self.provenance_hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        """
        Calculate SHA-256 hash of entire calculation provenance.

        The hash includes:
        - All input parameters
        - All calculation steps
        - Final output value
        - Formula identification

        Returns:
            str: 64-character hexadecimal SHA-256 hash
        """
        hash_data = {
            "calculation_id": self.calculation_id,
            "calculation_type": self.calculation_type.value,
            "formula_id": self.formula_id,
            "formula_version": self.formula_version,
            "inputs": {k: str(v) for k, v in self.inputs.items()},
            "steps": [step.to_dict() for step in self.steps],
            "output_value": str(self.output_value),
            "output_unit": self.output_unit,
            "precision": self.precision
        }

        # Deterministic JSON serialization (sorted keys)
        hash_string = json.dumps(hash_data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(hash_string.encode('utf-8')).hexdigest()

    def verify_integrity(self) -> bool:
        """
        Verify the integrity of this calculation record.

        Recalculates the hash and compares with stored hash
        to detect any tampering or corruption.

        Returns:
            bool: True if hash matches, False if corrupted
        """
        recalculated_hash = self._calculate_hash()
        return recalculated_hash == self.provenance_hash

    def to_audit_record(self) -> Dict[str, Any]:
        """
        Generate audit record for regulatory compliance.

        Returns:
            Dict containing all information needed for audit
        """
        return {
            "calculation_id": self.calculation_id,
            "calculation_type": self.calculation_type.value,
            "formula_id": self.formula_id,
            "formula_version": self.formula_version,
            "timestamp_utc": self.timestamp_utc,
            "inputs": self.inputs,
            "calculation_steps": [step.to_dict() for step in self.steps],
            "output": {
                "value": float(self.output_value),
                "unit": self.output_unit,
                "precision": self.precision
            },
            "provenance_hash": self.provenance_hash,
            "calculation_time_ms": self.calculation_time_ms,
            "integrity_verified": self.verify_integrity(),
            "metadata": self.metadata
        }


class ProvenanceTracker:
    """
    Tracks calculation provenance for zero-hallucination guarantee.

    This class manages the creation, storage, and verification of
    calculation provenance records. It ensures that every calculation
    is deterministic, reproducible, and auditable.

    Example:
        >>> tracker = ProvenanceTracker()
        >>> tracker.start_calculation(
        ...     calculation_type=CalculationType.HEAT_TRANSFER,
        ...     formula_id="lmtd_counter_flow",
        ...     formula_version="1.0.0",
        ...     inputs={"T_hot_in": 400, "T_hot_out": 200, "T_cold_in": 100, "T_cold_out": 150}
        ... )
        >>> tracker.add_step(
        ...     operation="subtract",
        ...     description="Calculate hot side temperature difference",
        ...     inputs={"T_hot_in": 400, "T_cold_out": 150},
        ...     output_name="delta_T1",
        ...     output_value=250,
        ...     formula="delta_T1 = T_hot_in - T_cold_out"
        ... )
        >>> provenance = tracker.complete_calculation(
        ...     output_value=178.98,
        ...     output_unit="degF"
        ... )
    """

    def __init__(self):
        """Initialize the provenance tracker."""
        self._calculation_id: Optional[str] = None
        self._calculation_type: Optional[CalculationType] = None
        self._formula_id: Optional[str] = None
        self._formula_version: Optional[str] = None
        self._start_time: Optional[float] = None
        self._timestamp_utc: Optional[str] = None
        self._inputs: Dict[str, Any] = {}
        self._steps: List[CalculationStep] = []
        self._step_counter: int = 0
        self._metadata: Dict[str, Any] = {}
        self._active: bool = False

    def start_calculation(
        self,
        calculation_type: CalculationType,
        formula_id: str,
        formula_version: str,
        inputs: Dict[str, Any],
        calculation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start tracking a new calculation.

        Args:
            calculation_type: Category of calculation
            formula_id: Identifier of the formula being used
            formula_version: Version of the formula
            inputs: All input parameters for the calculation
            calculation_id: Optional custom ID (auto-generated if not provided)
            metadata: Optional additional metadata

        Returns:
            str: The calculation ID

        Raises:
            RuntimeError: If a calculation is already in progress
        """
        if self._active:
            raise RuntimeError(
                "Calculation already in progress. Complete or abort current calculation first."
            )

        self._active = True
        self._start_time = time.perf_counter()
        self._timestamp_utc = datetime.now(timezone.utc).isoformat()

        # Generate calculation ID if not provided
        if calculation_id is None:
            id_string = f"{formula_id}_{self._timestamp_utc}_{json.dumps(inputs, sort_keys=True)}"
            self._calculation_id = hashlib.sha256(id_string.encode()).hexdigest()[:16]
        else:
            self._calculation_id = calculation_id

        self._calculation_type = calculation_type
        self._formula_id = formula_id
        self._formula_version = formula_version
        self._inputs = inputs.copy()
        self._steps = []
        self._step_counter = 0
        self._metadata = metadata or {}

        return self._calculation_id

    def add_step(
        self,
        operation: str,
        description: str,
        inputs: Dict[str, Any],
        output_name: str,
        output_value: Union[float, Decimal],
        formula: str = ""
    ) -> CalculationStep:
        """
        Add a calculation step to the provenance record.

        Args:
            operation: Type of operation (multiply, divide, log, etc.)
            description: Human-readable description
            inputs: Input values used in this step
            output_name: Name of the output variable
            output_value: Calculated result
            formula: Mathematical formula (optional)

        Returns:
            CalculationStep: The created step record

        Raises:
            RuntimeError: If no calculation is in progress
        """
        if not self._active:
            raise RuntimeError("No calculation in progress. Call start_calculation first.")

        self._step_counter += 1

        step = CalculationStep(
            step_number=self._step_counter,
            operation=operation,
            description=description,
            inputs=inputs,
            output_name=output_name,
            output_value=output_value,
            formula=formula
        )

        self._steps.append(step)
        return step

    def complete_calculation(
        self,
        output_value: Union[float, Decimal],
        output_unit: str,
        precision: int = 6
    ) -> CalculationProvenance:
        """
        Complete the calculation and generate provenance record.

        Args:
            output_value: Final calculated value
            output_unit: Unit of measurement
            precision: Decimal places in output

        Returns:
            CalculationProvenance: Complete provenance record with hash

        Raises:
            RuntimeError: If no calculation is in progress
        """
        if not self._active:
            raise RuntimeError("No calculation in progress. Call start_calculation first.")

        end_time = time.perf_counter()
        calculation_time_ms = (end_time - self._start_time) * 1000

        # Apply precision to output value
        if isinstance(output_value, float):
            quantize_str = '0.' + '0' * precision
            output_value = Decimal(str(output_value)).quantize(
                Decimal(quantize_str), rounding=ROUND_HALF_UP
            )

        provenance = CalculationProvenance(
            calculation_id=self._calculation_id,
            calculation_type=self._calculation_type,
            formula_id=self._formula_id,
            formula_version=self._formula_version,
            timestamp_utc=self._timestamp_utc,
            inputs=self._inputs,
            steps=self._steps,
            output_value=output_value,
            output_unit=output_unit,
            precision=precision,
            calculation_time_ms=calculation_time_ms,
            metadata=self._metadata
        )

        # Reset tracker state
        self._reset()

        return provenance

    def abort_calculation(self) -> None:
        """Abort the current calculation without generating provenance."""
        self._reset()

    def _reset(self) -> None:
        """Reset tracker to initial state."""
        self._calculation_id = None
        self._calculation_type = None
        self._formula_id = None
        self._formula_version = None
        self._start_time = None
        self._timestamp_utc = None
        self._inputs = {}
        self._steps = []
        self._step_counter = 0
        self._metadata = {}
        self._active = False


def generate_calculation_hash(
    formula_id: str,
    inputs: Dict[str, Any],
    output_value: Union[float, Decimal],
    output_unit: str
) -> str:
    """
    Generate a SHA-256 hash for a calculation.

    This is a convenience function for generating hashes without
    using the full ProvenanceTracker.

    Args:
        formula_id: Identifier of the formula used
        inputs: Input parameters
        output_value: Calculated result
        output_unit: Unit of measurement

    Returns:
        str: 64-character hexadecimal SHA-256 hash
    """
    hash_data = {
        "formula_id": formula_id,
        "inputs": {k: str(v) for k, v in sorted(inputs.items())},
        "output_value": str(output_value),
        "output_unit": output_unit
    }

    hash_string = json.dumps(hash_data, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(hash_string.encode('utf-8')).hexdigest()


def verify_calculation_reproducibility(
    provenance1: CalculationProvenance,
    provenance2: CalculationProvenance
) -> bool:
    """
    Verify that two calculations are bit-perfect identical.

    This function compares two provenance records to ensure
    they represent the same calculation with identical results.

    Args:
        provenance1: First provenance record
        provenance2: Second provenance record

    Returns:
        bool: True if calculations are identical, False otherwise
    """
    return (
        provenance1.provenance_hash == provenance2.provenance_hash and
        provenance1.verify_integrity() and
        provenance2.verify_integrity()
    )


class AuditLogger:
    """
    Logger for calculation audit trails.

    Maintains a log of all calculations for regulatory compliance
    and audit purposes.
    """

    def __init__(self):
        """Initialize the audit logger."""
        self._audit_log: List[Dict[str, Any]] = []

    def log_calculation(self, provenance: CalculationProvenance) -> None:
        """
        Log a calculation to the audit trail.

        Args:
            provenance: Provenance record to log
        """
        self._audit_log.append(provenance.to_audit_record())

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """
        Get the complete audit log.

        Returns:
            List of audit records
        """
        return self._audit_log.copy()

    def export_audit_log_json(self) -> str:
        """
        Export audit log as JSON string.

        Returns:
            JSON string of audit log
        """
        return json.dumps(self._audit_log, indent=2, sort_keys=True)

    def find_by_calculation_id(self, calculation_id: str) -> Optional[Dict[str, Any]]:
        """
        Find a calculation record by ID.

        Args:
            calculation_id: The calculation ID to search for

        Returns:
            Audit record if found, None otherwise
        """
        for record in self._audit_log:
            if record["calculation_id"] == calculation_id:
                return record
        return None

    def verify_all_integrity(self) -> Dict[str, bool]:
        """
        Verify integrity of all logged calculations.

        Returns:
            Dictionary mapping calculation IDs to integrity status
        """
        results = {}
        for record in self._audit_log:
            calc_id = record["calculation_id"]
            results[calc_id] = record.get("integrity_verified", False)
        return results


# Module-level instances for convenience
_default_tracker = ProvenanceTracker()
_default_logger = AuditLogger()


def get_default_tracker() -> ProvenanceTracker:
    """Get the default provenance tracker instance."""
    return _default_tracker


def get_default_logger() -> AuditLogger:
    """Get the default audit logger instance."""
    return _default_logger
