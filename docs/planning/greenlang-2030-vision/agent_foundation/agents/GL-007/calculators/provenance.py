# -*- coding: utf-8 -*-
"""
Provenance Tracking for GL-007 FURNACEPULSE FurnacePerformanceMonitor

This module provides cryptographic provenance tracking for all furnace performance
calculations, ensuring complete audit trails, bit-perfect reproducibility, and
zero-hallucination guarantees for regulatory compliance.

Author: GL-CalculatorEngineer
Agent: GL-007 FURNACEPULSE
Version: 1.0.0
Standard: SHA-256 cryptographic hashing per GreenLang Zero-Hallucination Protocol
"""

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from decimal import Decimal, ROUND_HALF_UP
from dataclasses import dataclass, field, asdict
from enum import Enum


class CalculationCategory(Enum):
    """Categories of furnace performance calculations."""
    THERMAL_EFFICIENCY = "thermal_efficiency"
    HEAT_BALANCE = "heat_balance"
    WALL_LOSS = "wall_loss"
    FLUE_GAS_LOSS = "flue_gas_loss"
    PERFORMANCE_KPI = "performance_kpi"
    SPECIFIC_FUEL_CONSUMPTION = "specific_fuel_consumption"
    HEAT_RATE = "heat_rate"
    AVAILABILITY = "availability"
    UTILIZATION = "utilization"
    MAINTENANCE_PREDICTION = "maintenance_prediction"
    REFRACTORY_WEAR = "refractory_wear"
    BURNER_DEGRADATION = "burner_degradation"
    TREND_ANALYSIS = "trend_analysis"


@dataclass
class CalculationStep:
    """
    Represents a single step in a furnace performance calculation chain.

    Each step captures the operation performed, inputs used, and output produced,
    enabling complete traceability for audit purposes.

    Attributes:
        step_number: Sequential step number (1-indexed)
        operation: Mathematical operation performed (e.g., 'multiply', 'divide')
        description: Human-readable description of the step
        inputs: Dictionary of input values used in this step
        output_value: Result of the calculation step
        output_name: Name/identifier for the output value
        timestamp: ISO 8601 timestamp of when step was executed
        formula: Mathematical formula used (for documentation)
        units: Engineering units of the output value
        standard_reference: Reference to applicable standard (e.g., 'ASME PTC 4.2 Section 5.3')
    """
    step_number: int
    operation: str
    description: str
    inputs: Dict[str, Any]
    output_value: Any
    output_name: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    formula: Optional[str] = None
    units: Optional[str] = None
    standard_reference: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary with proper Decimal serialization.

        Returns:
            Dictionary representation with all Decimal values converted to strings
            for JSON compatibility while preserving precision.
        """
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
    """
    Complete provenance record for a furnace performance calculation.

    This record provides complete traceability for regulatory audits and
    third-party verification. Every calculation in GL-007 FURNACEPULSE
    produces a ProvenanceRecord that can be independently verified.

    Attributes:
        calculation_id: Unique identifier for this calculation instance
        calculation_type: Category of calculation (from CalculationCategory)
        version: Calculator version for reproducibility
        input_parameters: All input parameters used
        calculation_steps: Ordered list of all calculation steps
        final_result: Final output value
        provenance_hash: SHA-256 hash of entire calculation chain
        timestamp: When calculation was completed
        metadata: Additional context (furnace ID, operator, etc.)
        standard_compliance: List of standards this calculation complies with
        uncertainty_percent: Calculation uncertainty per applicable standard
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
    standard_compliance: List[str] = field(default_factory=list)
    uncertainty_percent: Optional[Decimal] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary representation suitable for JSON serialization
            or database storage.
        """
        data = asdict(self)
        data['calculation_steps'] = [step.to_dict() for step in self.calculation_steps]

        # Handle Decimal serialization
        if isinstance(data['final_result'], Decimal):
            data['final_result'] = str(data['final_result'])
        for key, value in data['input_parameters'].items():
            if isinstance(value, Decimal):
                data['input_parameters'][key] = str(value)
        if isinstance(data['uncertainty_percent'], Decimal):
            data['uncertainty_percent'] = str(data['uncertainty_percent'])
        return data

    def to_json(self) -> str:
        """
        Convert to JSON string for storage or transmission.

        Returns:
            Deterministic JSON string (sorted keys) for consistent hashing.
        """
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)


class ProvenanceTracker:
    """
    Tracks calculation provenance with SHA-256 hashing for GL-007 FURNACEPULSE.

    This tracker ensures every furnace performance calculation has complete
    audit trail capability with cryptographic integrity verification.

    Zero-Hallucination Guarantees:
    - Deterministic hashing: Same input always produces same hash
    - Complete audit trail: Every step is recorded
    - Tamper detection: Any modification invalidates the hash
    - NO LLM involvement: Pure mathematical provenance

    Example:
        >>> tracker = ProvenanceTracker(
        ...     calculation_id="furnace_eff_001",
        ...     calculation_type="thermal_efficiency",
        ...     version="1.0.0"
        ... )
        >>> tracker.record_inputs({'fuel_rate_kg_hr': 500, 'temp_c': 1200})
        >>> tracker.record_step(
        ...     operation="multiply",
        ...     description="Calculate heat input",
        ...     inputs={'fuel_rate': Decimal('500'), 'lhv': Decimal('45.5')},
        ...     output_value=Decimal('22750'),
        ...     output_name="heat_input_mj_hr"
        ... )
        >>> record = tracker.get_provenance_record(final_result=Decimal('82.5'))
    """

    def __init__(
        self,
        calculation_id: str,
        calculation_type: str,
        version: str = "1.0.0",
        standard_compliance: Optional[List[str]] = None
    ):
        """
        Initialize provenance tracker for a furnace performance calculation.

        Args:
            calculation_id: Unique identifier for this calculation
            calculation_type: Type of calculation (e.g., 'thermal_efficiency')
            version: Calculator version string
            standard_compliance: List of applicable standards
        """
        self.calculation_id = calculation_id
        self.calculation_type = calculation_type
        self.version = version
        self.standard_compliance = standard_compliance or ["ASME PTC 4.2"]
        self.steps: List[CalculationStep] = []
        self.input_parameters: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}

    def record_inputs(self, parameters: Dict[str, Any]) -> None:
        """
        Record input parameters for the calculation.

        All inputs are normalized to Decimal for precision and reproducibility.

        Args:
            parameters: Dictionary of input parameter names and values
        """
        self.input_parameters = self._normalize_values(parameters.copy())

    def set_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Set additional metadata for the calculation.

        Args:
            metadata: Dictionary of metadata (furnace_id, operator, etc.)
        """
        self.metadata = metadata.copy()

    def record_step(
        self,
        operation: str,
        description: str,
        inputs: Dict[str, Any],
        output_value: Any,
        output_name: str,
        formula: Optional[str] = None,
        units: Optional[str] = None,
        standard_reference: Optional[str] = None
    ) -> CalculationStep:
        """
        Record a single calculation step with full provenance.

        Each step captures the exact operation, inputs, and outputs for
        complete audit trail capability.

        Args:
            operation: Type of operation ('multiply', 'divide', 'add', etc.)
            description: Human-readable description of what this step does
            inputs: Dictionary of input values for this step
            output_value: Result of the calculation
            output_name: Identifier for the output
            formula: Mathematical formula (e.g., 'Q = m * Cp * dT')
            units: Engineering units of output (e.g., 'kW', 'MJ/hr')
            standard_reference: Reference to applicable standard section

        Returns:
            The recorded CalculationStep for reference
        """
        step = CalculationStep(
            step_number=len(self.steps) + 1,
            operation=operation,
            description=description,
            inputs=self._normalize_values(inputs),
            output_value=self._normalize_value(output_value),
            output_name=output_name,
            formula=formula,
            units=units,
            standard_reference=standard_reference
        )
        self.steps.append(step)
        return step

    def generate_hash(self, final_result: Any) -> str:
        """
        Generate SHA-256 hash of the complete calculation.

        This hash is DETERMINISTIC - same inputs and calculation steps
        will ALWAYS produce the identical hash, guaranteeing bit-perfect
        reproducibility for regulatory audits.

        Args:
            final_result: The final output value of the calculation

        Returns:
            64-character hexadecimal SHA-256 hash string
        """
        # Create canonical representation for hashing
        canonical_data = {
            'calculation_id': self.calculation_id,
            'calculation_type': self.calculation_type,
            'version': self.version,
            'input_parameters': self._serialize_for_hash(self.input_parameters),
            'steps': [self._serialize_step_for_hash(step) for step in self.steps],
            'final_result': self._serialize_for_hash(final_result)
        }

        # Generate deterministic JSON string (sorted keys, no whitespace)
        canonical_json = json.dumps(canonical_data, sort_keys=True, separators=(',', ':'))

        # Calculate SHA-256 hash
        return hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()

    def get_provenance_record(
        self,
        final_result: Any,
        uncertainty_percent: Optional[Decimal] = None
    ) -> ProvenanceRecord:
        """
        Get complete provenance record for the calculation.

        Args:
            final_result: The final output value
            uncertainty_percent: Calculation uncertainty (per ASME PTC 4.2)

        Returns:
            Complete ProvenanceRecord with hash and all steps
        """
        provenance_hash = self.generate_hash(final_result)

        return ProvenanceRecord(
            calculation_id=self.calculation_id,
            calculation_type=self.calculation_type,
            version=self.version,
            input_parameters=self.input_parameters,
            calculation_steps=self.steps,
            final_result=self._normalize_value(final_result),
            provenance_hash=provenance_hash,
            metadata=self.metadata,
            standard_compliance=self.standard_compliance,
            uncertainty_percent=uncertainty_percent
        )

    def _normalize_value(self, value: Any) -> Any:
        """
        Normalize a single value for consistent handling.

        Converts floats and ints to Decimal for precision.
        """
        if isinstance(value, (int, float)) and not isinstance(value, bool):
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
        """
        Serialize values for deterministic hash calculation.

        All numeric values are converted to string representations
        to ensure consistent hashing across platforms.
        """
        if isinstance(value, Decimal):
            return str(value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(Decimal(str(value)))
        elif isinstance(value, dict):
            return {k: self._serialize_for_hash(v) for k, v in sorted(value.items())}
        elif isinstance(value, list):
            return [self._serialize_for_hash(v) for v in value]
        elif value is None:
            return 'null'
        return str(value)

    def _serialize_step_for_hash(self, step: CalculationStep) -> Dict[str, Any]:
        """Serialize a calculation step for deterministic hashing."""
        return {
            'step_number': step.step_number,
            'operation': step.operation,
            'description': step.description,
            'inputs': self._serialize_for_hash(step.inputs),
            'output_value': self._serialize_for_hash(step.output_value),
            'output_name': step.output_name,
            'formula': step.formula or '',
            'units': step.units or '',
            'standard_reference': step.standard_reference or ''
        }


class ProvenanceValidator:
    """
    Validates calculation provenance for integrity and reproducibility.

    Used by auditors and compliance systems to verify that calculations
    have not been tampered with and are reproducible.
    """

    @staticmethod
    def validate_hash(record: ProvenanceRecord) -> bool:
        """
        Validate that a provenance hash matches the calculation data.

        This verifies the calculation has not been tampered with since
        the hash was generated.

        Args:
            record: ProvenanceRecord to validate

        Returns:
            True if hash is valid (calculation not tampered), False otherwise
        """
        # Recreate the tracker from record data
        tracker = ProvenanceTracker(
            record.calculation_id,
            record.calculation_type,
            record.version,
            record.standard_compliance
        )
        tracker.input_parameters = record.input_parameters
        tracker.steps = record.calculation_steps

        # Recalculate hash
        expected_hash = tracker.generate_hash(record.final_result)

        return expected_hash == record.provenance_hash

    @staticmethod
    def validate_reproducibility(
        record1: ProvenanceRecord,
        record2: ProvenanceRecord
    ) -> bool:
        """
        Validate that two calculations with same inputs produce same results.

        This verifies the zero-hallucination guarantee - deterministic
        calculations must always produce identical results.

        Args:
            record1: First calculation record
            record2: Second calculation record

        Returns:
            True if both calculations are identical (reproducible)
        """
        # Check calculation types match
        if record1.calculation_type != record2.calculation_type:
            return False

        # Check inputs match (comparing serialized forms)
        if str(record1.input_parameters) != str(record2.input_parameters):
            return False

        # Check results match (bit-perfect)
        if str(record1.final_result) != str(record2.final_result):
            return False

        # Check hashes match
        return record1.provenance_hash == record2.provenance_hash

    @staticmethod
    def validate_compliance(
        record: ProvenanceRecord,
        required_standard: str
    ) -> bool:
        """
        Validate that calculation complies with a required standard.

        Args:
            record: ProvenanceRecord to validate
            required_standard: Standard that must be in compliance list

        Returns:
            True if calculation claims compliance with the standard
        """
        return required_standard in record.standard_compliance


def create_calculation_hash(data: Dict[str, Any]) -> str:
    """
    Create a deterministic SHA-256 hash for any calculation data.

    Utility function for quick hashing of arbitrary data structures.

    Args:
        data: Dictionary of data to hash

    Returns:
        64-character hexadecimal SHA-256 hash string

    Example:
        >>> hash_value = create_calculation_hash({
        ...     'efficiency': 82.5,
        ...     'fuel_rate': 500
        ... })
    """
    # Serialize with sorted keys for determinism
    canonical_json = json.dumps(
        _serialize_dict_for_hash(data),
        sort_keys=True,
        separators=(',', ':')
    )
    return hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()


def _serialize_dict_for_hash(data: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively serialize dictionary for hashing."""
    result = {}
    for key, value in data.items():
        if isinstance(value, Decimal):
            result[key] = str(value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            result[key] = str(Decimal(str(value)))
        elif isinstance(value, dict):
            result[key] = _serialize_dict_for_hash(value)
        elif isinstance(value, list):
            result[key] = [
                _serialize_dict_for_hash(v) if isinstance(v, dict)
                else str(Decimal(str(v))) if isinstance(v, (int, float)) and not isinstance(v, bool)
                else str(v) if isinstance(v, Decimal)
                else v
                for v in value
            ]
        elif value is None:
            result[key] = 'null'
        else:
            result[key] = str(value)
    return result


def verify_calculation_integrity(
    calculation_data: Dict[str, Any],
    expected_hash: str
) -> bool:
    """
    Verify that calculation data matches an expected hash.

    Used by external systems to verify calculation integrity.

    Args:
        calculation_data: The calculation data to verify
        expected_hash: The expected SHA-256 hash

    Returns:
        True if data matches the expected hash
    """
    actual_hash = create_calculation_hash(calculation_data)
    return actual_hash == expected_hash
