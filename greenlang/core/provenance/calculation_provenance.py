# -*- coding: utf-8 -*-
"""
GreenLang Calculation Provenance - Standardized Audit Trail Tracking

This module provides standardized provenance tracking for all GreenLang calculators,
based on CSRD, CBAM, and GL-001 through GL-010 best practices.

Features:
- Step-by-step calculation recording
- SHA-256 hash-based integrity verification
- Complete audit trail with timestamps
- Standard reference tracking (EPA, ISO, ASME, etc.)
- Data source lineage
- Zero-hallucination guarantees

Author: GreenLang Team
Version: 1.0.0
Standards: CSRD, CBAM, ISO 14064, GHG Protocol
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from decimal import Decimal
import hashlib
import json
from enum import Enum

# Import determinism utilities
try:
    from greenlang.determinism import (
        DeterministicClock,
        content_hash,
        safe_decimal,
    )
except ImportError:
    # Fallback for testing
    from datetime import timezone

    class DeterministicClock:
        @staticmethod
        def now():
            return datetime.now(timezone.utc).replace(microsecond=0)

    def content_hash(data: Any) -> str:
        if isinstance(data, dict):
            data = json.dumps(data, sort_keys=True)
        if isinstance(data, str):
            data = data.encode('utf-8')
        return hashlib.sha256(data).hexdigest()

    def safe_decimal(value: Any) -> Decimal:
        return Decimal(str(value))


class OperationType(str, Enum):
    """Standard operation types for calculations."""
    LOOKUP = "lookup"
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    AGGREGATE = "aggregate"
    CONVERT = "convert"
    VALIDATE = "validate"
    TRANSFORM = "transform"
    INTERPOLATE = "interpolate"
    EXTRAPOLATE = "extrapolate"


@dataclass
class CalculationStep:
    """
    Single step in a calculation with complete provenance.

    Each step represents a discrete calculation operation with full
    traceability for regulatory compliance and audit requirements.

    Attributes:
        step_number: Sequential step number (1-based)
        operation: Type of operation performed
        description: Human-readable description of what this step does
        inputs: Input values used in this step
        output: Output value produced by this step
        formula: Mathematical formula (optional, e.g., "emissions = activity_data * emission_factor")
        data_source: Source of data (optional, e.g., "DEFRA 2024", "Supplier Invoice #123")
        standard_reference: Standard/regulation reference (optional, e.g., "ISO 14064-1", "EPA AP-42")
        timestamp: When this step was executed (ISO 8601)
        metadata: Additional metadata for this step

    Example:
        >>> step = CalculationStep(
        ...     step_number=1,
        ...     operation=OperationType.LOOKUP,
        ...     description="Lookup emission factor for natural gas",
        ...     inputs={"fuel_type": "natural_gas", "region": "US"},
        ...     output=0.18414,
        ...     data_source="EPA eGRID 2023",
        ...     standard_reference="EPA AP-42"
        ... )
    """
    step_number: int
    operation: Union[OperationType, str]
    description: str
    inputs: Dict[str, Any]
    output: Any
    formula: Optional[str] = None
    data_source: Optional[str] = None
    standard_reference: Optional[str] = None
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = DeterministicClock.now().isoformat()

        # Convert operation to string if enum
        if isinstance(self.operation, OperationType):
            self.operation = self.operation.value

    def to_dict(self) -> Dict[str, Any]:
        """Export step to dictionary for serialization."""
        return {
            "step_number": self.step_number,
            "operation": self.operation,
            "description": self.description,
            "inputs": self.inputs,
            "output": self.output,
            "formula": self.formula,
            "data_source": self.data_source,
            "standard_reference": self.standard_reference,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CalculationStep':
        """Create step from dictionary."""
        return cls(
            step_number=data["step_number"],
            operation=data["operation"],
            description=data["description"],
            inputs=data["inputs"],
            output=data["output"],
            formula=data.get("formula"),
            data_source=data.get("data_source"),
            standard_reference=data.get("standard_reference"),
            timestamp=data.get("timestamp"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ProvenanceMetadata:
    """
    Metadata for provenance tracking.

    Attributes:
        agent_name: Name of the agent performing calculation
        agent_version: Version of the agent
        calculation_type: Type of calculation (e.g., "emissions", "efficiency")
        standards_applied: List of standards/regulations applied
        data_sources: List of data sources used
        warnings: List of warning messages
        errors: List of error messages
        custom: Additional custom metadata
    """
    agent_name: str
    agent_version: str
    calculation_type: str
    standards_applied: List[str] = field(default_factory=list)
    data_sources: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    custom: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Export metadata to dictionary."""
        return {
            "agent_name": self.agent_name,
            "agent_version": self.agent_version,
            "calculation_type": self.calculation_type,
            "standards_applied": self.standards_applied,
            "data_sources": self.data_sources,
            "warnings": self.warnings,
            "errors": self.errors,
            "custom": self.custom,
        }


@dataclass
class CalculationProvenance:
    """
    Complete provenance record for a calculation.

    This class provides standardized audit trail tracking for all GreenLang
    calculators, ensuring zero-hallucination calculations with complete
    traceability.

    Attributes:
        calculation_id: Unique ID for this calculation
        metadata: Provenance metadata (agent, version, type, etc.)
        input_data: All input values
        input_hash: SHA-256 hash of inputs (for deduplication)
        steps: Ordered list of calculation steps
        output_data: Final result
        output_hash: SHA-256 hash of output
        timestamp_start: When calculation started (ISO 8601)
        timestamp_end: When calculation ended (ISO 8601)
        duration_ms: Duration in milliseconds

    Example:
        >>> # Create provenance record
        >>> provenance = CalculationProvenance.create(
        ...     agent_name="EmissionsCalculator",
        ...     agent_version="1.0.0",
        ...     calculation_type="scope1_emissions",
        ...     input_data={"fuel_consumption_kg": 1000, "fuel_type": "natural_gas"}
        ... )
        >>>
        >>> # Add calculation steps
        >>> provenance.add_step(
        ...     operation=OperationType.LOOKUP,
        ...     description="Lookup emission factor",
        ...     inputs={"fuel_type": "natural_gas"},
        ...     output=0.18414,
        ...     data_source="EPA eGRID 2023"
        ... )
        >>>
        >>> provenance.add_step(
        ...     operation=OperationType.MULTIPLY,
        ...     description="Calculate total emissions",
        ...     inputs={"fuel_consumption_kg": 1000, "emission_factor": 0.18414},
        ...     output=184.14,
        ...     formula="emissions = fuel_consumption * emission_factor"
        ... )
        >>>
        >>> # Finalize with output
        >>> provenance.finalize(output_data={"total_emissions_kg_co2e": 184.14})
        >>>
        >>> # Export for storage
        >>> record = provenance.to_dict()
    """
    calculation_id: str
    metadata: ProvenanceMetadata
    input_data: Dict[str, Any]
    input_hash: str
    steps: List[CalculationStep] = field(default_factory=list)
    output_data: Optional[Any] = None
    output_hash: Optional[str] = None
    timestamp_start: Optional[str] = None
    timestamp_end: Optional[str] = None
    duration_ms: float = 0.0

    @classmethod
    def create(
        cls,
        agent_name: str,
        agent_version: str,
        calculation_type: str,
        input_data: Dict[str, Any],
        standards_applied: Optional[List[str]] = None,
        data_sources: Optional[List[str]] = None,
    ) -> 'CalculationProvenance':
        """
        Create new provenance record.

        Args:
            agent_name: Name of the agent performing calculation
            agent_version: Version of the agent
            calculation_type: Type of calculation (e.g., "emissions", "efficiency")
            input_data: All input values
            standards_applied: List of standards/regulations applied
            data_sources: List of data sources used

        Returns:
            New CalculationProvenance instance

        Example:
            >>> provenance = CalculationProvenance.create(
            ...     agent_name="CarbonCalculator",
            ...     agent_version="2.1.0",
            ...     calculation_type="scope2_emissions",
            ...     input_data={"electricity_kwh": 5000, "region": "US-CA"},
            ...     standards_applied=["GHG Protocol Scope 2"],
            ...     data_sources=["EPA eGRID 2023"]
            ... )
        """
        # Generate calculation ID from inputs and timestamp
        timestamp_start = DeterministicClock.now().isoformat()
        calc_id_data = {
            "agent": agent_name,
            "timestamp": timestamp_start,
            "input": input_data
        }
        calculation_id = content_hash(calc_id_data)[:16]  # Use first 16 chars

        # Create metadata
        metadata = ProvenanceMetadata(
            agent_name=agent_name,
            agent_version=agent_version,
            calculation_type=calculation_type,
            standards_applied=standards_applied or [],
            data_sources=data_sources or [],
        )

        # Hash input data
        input_hash = content_hash(input_data)

        return cls(
            calculation_id=calculation_id,
            metadata=metadata,
            input_data=input_data,
            input_hash=input_hash,
            steps=[],
            output_data=None,
            output_hash=None,
            timestamp_start=timestamp_start,
            timestamp_end=None,
            duration_ms=0.0,
        )

    def add_step(
        self,
        operation: Union[OperationType, str],
        description: str,
        inputs: Dict[str, Any],
        output: Any,
        formula: Optional[str] = None,
        data_source: Optional[str] = None,
        standard_reference: Optional[str] = None,
        **kwargs
    ) -> CalculationStep:
        """
        Add a calculation step.

        Args:
            operation: Type of operation (lookup, add, multiply, etc.)
            description: Human-readable description
            inputs: Input values for this step
            output: Output value from this step
            formula: Mathematical formula (optional)
            data_source: Source of data (optional)
            standard_reference: Standard/regulation reference (optional)
            **kwargs: Additional metadata

        Returns:
            The created CalculationStep

        Example:
            >>> step = provenance.add_step(
            ...     operation=OperationType.DIVIDE,
            ...     description="Calculate efficiency ratio",
            ...     inputs={"useful_output": 850, "energy_input": 1000},
            ...     output=0.85,
            ...     formula="efficiency = useful_output / energy_input",
            ...     standard_reference="ISO 50001"
            ... )
        """
        step = CalculationStep(
            step_number=len(self.steps) + 1,
            operation=operation,
            description=description,
            inputs=inputs,
            output=output,
            formula=formula,
            data_source=data_source,
            standard_reference=standard_reference,
            metadata=kwargs,
        )
        self.steps.append(step)

        # Update data sources if provided
        if data_source and data_source not in self.metadata.data_sources:
            self.metadata.data_sources.append(data_source)

        # Update standards if provided
        if standard_reference and standard_reference not in self.metadata.standards_applied:
            self.metadata.standards_applied.append(standard_reference)

        return step

    def add_warning(self, warning: str):
        """
        Add a warning message.

        Args:
            warning: Warning message

        Example:
            >>> provenance.add_warning("Using default emission factor due to missing region data")
        """
        self.metadata.warnings.append(warning)

    def add_error(self, error: str):
        """
        Add an error message.

        Args:
            error: Error message

        Example:
            >>> provenance.add_error("Division by zero in efficiency calculation")
        """
        self.metadata.errors.append(error)

    def finalize(self, output_data: Any):
        """
        Finalize provenance record with output data.

        This method should be called after all calculation steps are complete.
        It calculates the output hash and duration.

        Args:
            output_data: Final calculation result

        Example:
            >>> provenance.finalize(output_data={
            ...     "total_emissions_kg_co2e": 184.14,
            ...     "emissions_per_unit": 0.18414
            ... })
        """
        self.output_data = output_data
        # Handle None output_data
        if output_data is None:
            self.output_hash = content_hash("")
        else:
            self.output_hash = content_hash(output_data)
        self.timestamp_end = DeterministicClock.now().isoformat()

        # Calculate duration
        if self.timestamp_start and self.timestamp_end:
            start = datetime.fromisoformat(self.timestamp_start)
            end = datetime.fromisoformat(self.timestamp_end)
            self.duration_ms = (end - start).total_seconds() * 1000

    def to_dict(self) -> Dict[str, Any]:
        """
        Export to dictionary for storage.

        Returns:
            Complete provenance record as dictionary

        Example:
            >>> record_dict = provenance.to_dict()
            >>> import json
            >>> with open("provenance.json", "w") as f:
            ...     json.dump(record_dict, f, indent=2)
        """
        return {
            "calculation_id": self.calculation_id,
            "metadata": self.metadata.to_dict(),
            "input_data": self.input_data,
            "input_hash": self.input_hash,
            "steps": [step.to_dict() for step in self.steps],
            "output_data": self.output_data,
            "output_hash": self.output_hash,
            "timestamp_start": self.timestamp_start,
            "timestamp_end": self.timestamp_end,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CalculationProvenance':
        """
        Create provenance record from dictionary.

        Args:
            data: Dictionary representation of provenance record

        Returns:
            CalculationProvenance instance

        Example:
            >>> with open("provenance.json") as f:
            ...     data = json.load(f)
            >>> provenance = CalculationProvenance.from_dict(data)
        """
        metadata = ProvenanceMetadata(
            agent_name=data["metadata"]["agent_name"],
            agent_version=data["metadata"]["agent_version"],
            calculation_type=data["metadata"]["calculation_type"],
            standards_applied=data["metadata"].get("standards_applied", []),
            data_sources=data["metadata"].get("data_sources", []),
            warnings=data["metadata"].get("warnings", []),
            errors=data["metadata"].get("errors", []),
            custom=data["metadata"].get("custom", {}),
        )

        steps = [CalculationStep.from_dict(step) for step in data["steps"]]

        return cls(
            calculation_id=data["calculation_id"],
            metadata=metadata,
            input_data=data["input_data"],
            input_hash=data["input_hash"],
            steps=steps,
            output_data=data.get("output_data"),
            output_hash=data.get("output_hash"),
            timestamp_start=data.get("timestamp_start"),
            timestamp_end=data.get("timestamp_end"),
            duration_ms=data.get("duration_ms", 0.0),
        )

    def verify_input_hash(self) -> bool:
        """
        Verify input data hash integrity.

        Returns:
            True if hash matches, False otherwise

        Example:
            >>> assert provenance.verify_input_hash()
        """
        computed_hash = content_hash(self.input_data)
        return computed_hash == self.input_hash

    def verify_output_hash(self) -> bool:
        """
        Verify output data hash integrity.

        Returns:
            True if hash matches, False otherwise

        Example:
            >>> assert provenance.verify_output_hash()
        """
        if self.output_data is None or self.output_hash is None:
            return False
        computed_hash = content_hash(self.output_data)
        return computed_hash == self.output_hash

    def verify_integrity(self) -> Dict[str, bool]:
        """
        Verify complete integrity of provenance record.

        Returns:
            Dictionary with verification results

        Example:
            >>> integrity = provenance.verify_integrity()
            >>> assert integrity["input_hash_valid"]
            >>> assert integrity["output_hash_valid"]
            >>> assert integrity["steps_sequential"]
        """
        return {
            "input_hash_valid": self.verify_input_hash(),
            "output_hash_valid": self.verify_output_hash(),
            "steps_sequential": self._verify_steps_sequential(),
            "has_steps": len(self.steps) > 0,
            "is_finalized": self.output_data is not None,
            "no_errors": len(self.metadata.errors) == 0,
        }

    def _verify_steps_sequential(self) -> bool:
        """Verify steps are numbered sequentially starting from 1."""
        if not self.steps:
            return True
        expected = 1
        for step in self.steps:
            if step.step_number != expected:
                return False
            expected += 1
        return True

    def get_audit_summary(self) -> Dict[str, Any]:
        """
        Get human-readable audit summary.

        Returns:
            Summary of calculation for audit purposes

        Example:
            >>> summary = provenance.get_audit_summary()
            >>> print(summary["summary"])
            "EmissionsCalculator v1.0.0 performed scope1_emissions calculation in 12.5ms"
        """
        return {
            "calculation_id": self.calculation_id,
            "agent": f"{self.metadata.agent_name} v{self.metadata.agent_version}",
            "calculation_type": self.metadata.calculation_type,
            "timestamp": self.timestamp_start,
            "duration_ms": self.duration_ms,
            "steps_count": len(self.steps),
            "standards_applied": self.metadata.standards_applied,
            "data_sources": self.metadata.data_sources,
            "warnings_count": len(self.metadata.warnings),
            "errors_count": len(self.metadata.errors),
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "summary": (
                f"{self.metadata.agent_name} v{self.metadata.agent_version} "
                f"performed {self.metadata.calculation_type} calculation "
                f"in {self.duration_ms:.1f}ms with {len(self.steps)} steps"
            ),
        }


def stable_hash(data: Any) -> str:
    """
    Create stable hash for any data structure.

    This is a convenience function that wraps content_hash for
    backward compatibility with existing code.

    Args:
        data: Data to hash

    Returns:
        SHA-256 hash as hex string

    Example:
        >>> hash1 = stable_hash({"a": 1, "b": 2})
        >>> hash2 = stable_hash({"b": 2, "a": 1})  # Same despite different order
        >>> assert hash1 == hash2
    """
    return content_hash(data)
