"""
Provenance Tracking Utilities for Zero-Hallucination Calculations

This module provides cryptographic provenance tracking for all calculations,
ensuring complete audit trails and bit-perfect reproducibility.

Author: GL-CalculatorEngineer
Version: 1.0.0
Standard: SHA-256 cryptographic hashing
"""

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from decimal import Decimal
from dataclasses import dataclass, field, asdict
import traceback


@dataclass
class CalculationStep:
    """Represents a single step in a calculation chain."""
    step_number: int
    operation: str
    description: str
    inputs: Dict[str, Any]
    output_value: Any
    output_name: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    formula: Optional[str] = None
    units: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary with proper serialization."""
        data = asdict(self)
        # Convert Decimal values to string for JSON serialization
        for key, value in data['inputs'].items():
            if isinstance(value, Decimal):
                data['inputs'][key] = str(value)
        if isinstance(data['output_value'], Decimal):
            data['output_value'] = str(data['output_value'])
        return data


@dataclass
class ProvenanceRecord:
    """Complete provenance record for a calculation."""
    calculation_id: str
    calculation_type: str
    version: str
    input_parameters: Dict[str, Any]
    calculation_steps: List[CalculationStep]
    final_result: Any
    provenance_hash: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['calculation_steps'] = [step.to_dict() for step in self.calculation_steps]
        # Handle Decimal serialization
        if isinstance(data['final_result'], Decimal):
            data['final_result'] = str(data['final_result'])
        for key, value in data['input_parameters'].items():
            if isinstance(value, Decimal):
                data['input_parameters'][key] = str(value)
        return data


class ProvenanceTracker:
    """
    Tracks calculation provenance with SHA-256 hashing.

    Guarantees:
    - Deterministic hashing (same input → same hash)
    - Complete audit trail
    - Tamper detection
    - Zero hallucination (no LLM involvement)
    """

    def __init__(self, calculation_id: str, calculation_type: str, version: str = "1.0.0"):
        """Initialize provenance tracker."""
        self.calculation_id = calculation_id
        self.calculation_type = calculation_type
        self.version = version
        self.steps: List[CalculationStep] = []
        self.input_parameters: Dict[str, Any] = {}

    def record_inputs(self, parameters: Dict[str, Any]) -> None:
        """Record input parameters for the calculation."""
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
        """Record a calculation step."""
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

        # Generate deterministic JSON string (sorted keys)
        canonical_json = json.dumps(canonical_data, sort_keys=True, separators=(',', ':'))

        # Calculate SHA-256 hash
        return hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()

    def get_provenance_record(self, final_result: Any) -> ProvenanceRecord:
        """Get complete provenance record."""
        provenance_hash = self.generate_hash(final_result)

        return ProvenanceRecord(
            calculation_id=self.calculation_id,
            calculation_type=self.calculation_type,
            version=self.version,
            input_parameters=self.input_parameters,
            calculation_steps=self.steps,
            final_result=self._normalize_value(final_result),
            provenance_hash=provenance_hash
        )

    def _normalize_value(self, value: Any) -> Any:
        """Normalize values for consistent handling."""
        if isinstance(value, (int, float)):
            return Decimal(str(value))
        elif isinstance(value, dict):
            return {k: self._normalize_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._normalize_value(v) for v in value]
        return value

    def _normalize_values(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a dictionary of values."""
        return {k: self._normalize_value(v) for k, v in values.items()}

    def _serialize_for_hash(self, value: Any) -> Any:
        """Serialize values for hash calculation."""
        if isinstance(value, Decimal):
            return str(value)
        elif isinstance(value, (int, float)):
            return str(Decimal(str(value)))
        elif isinstance(value, dict):
            return {k: self._serialize_for_hash(v) for k, v in sorted(value.items())}
        elif isinstance(value, list):
            return [self._serialize_for_hash(v) for v in value]
        return str(value)

    def _serialize_step_for_hash(self, step: CalculationStep) -> Dict:
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


class ProvenanceValidator:
    """Validates calculation provenance for integrity."""

    @staticmethod
    def validate_hash(record: ProvenanceRecord) -> bool:
        """
        Validate that a provenance hash matches the calculation.

        Returns True if the hash is valid (calculation not tampered).
        """
        # Recreate the hash from the record data
        tracker = ProvenanceTracker(
            record.calculation_id,
            record.calculation_type,
            record.version
        )
        tracker.input_parameters = record.input_parameters
        tracker.steps = record.calculation_steps

        expected_hash = tracker.generate_hash(record.final_result)
        return expected_hash == record.provenance_hash

    @staticmethod
    def validate_reproducibility(
        record1: ProvenanceRecord,
        record2: ProvenanceRecord
    ) -> bool:
        """
        Validate that two calculations with same inputs produce same results.

        This verifies the zero-hallucination guarantee.
        """
        # Check inputs match
        if record1.input_parameters != record2.input_parameters:
            return False

        # Check calculation types match
        if record1.calculation_type != record2.calculation_type:
            return False

        # Check results match (bit-perfect)
        if str(record1.final_result) != str(record2.final_result):
            return False

        # Check hashes match
        return record1.provenance_hash == record2.provenance_hash


def create_calculation_hash(data: Dict[str, Any]) -> str:
    """
    Create a deterministic hash for any calculation data.

    Utility function for quick hashing needs.
    """
    canonical_json = json.dumps(data, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()