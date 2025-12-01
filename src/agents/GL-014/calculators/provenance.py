"""
Provenance Tracking Library for GL-014 EXCHANGER-PRO

This module provides comprehensive audit trail tracking for all
heat exchanger calculations. Implements SHA-256 hashing for
data integrity verification and complete calculation chain tracking.

Features:
- Step-by-step calculation recording with timestamps
- SHA-256 hash generation for audit trails
- Calculation chain dependency tracking
- JSON serialization for storage
- Compliance with regulatory audit requirements

Example:
    >>> from provenance import ProvenanceBuilder
    >>> provenance = ProvenanceBuilder(calculation_id="HX-001-RATING")
    >>> provenance.add_step("input_validation", {"tube_od": 0.019, "tube_length": 6.0})
    >>> provenance.add_step("htc_calculation", {"h_tube": 5000, "h_shell": 2500})
    >>> audit_record = provenance.finalize()
    >>> print(audit_record.provenance_hash)
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import logging
import uuid

logger = logging.getLogger(__name__)


# =============================================================================
# PROVENANCE DATA STRUCTURES
# =============================================================================

class CalculationStatus(Enum):
    """Status of a calculation step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ProvenanceLevel(Enum):
    """Level of provenance detail."""
    MINIMAL = "minimal"  # Only final results
    STANDARD = "standard"  # Key intermediate steps
    DETAILED = "detailed"  # All steps with full data
    DEBUG = "debug"  # Everything including internal state


@dataclass
class CalculationStep:
    """
    Represents a single step in a calculation chain.

    Attributes:
        step_id: Unique identifier for this step
        step_name: Human-readable name of the step
        timestamp: When the step was executed
        inputs: Input data for this step
        outputs: Output data from this step
        formula_used: Formula or method used
        references: References to standards or sources
        dependencies: List of step_ids this step depends on
        status: Current status of the step
        duration_ms: Execution time in milliseconds
        notes: Additional notes or warnings
    """
    step_id: str
    step_name: str
    timestamp: datetime
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    formula_used: Optional[str] = None
    references: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    status: CalculationStatus = CalculationStatus.COMPLETED
    duration_ms: float = 0.0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "step_id": self.step_id,
            "step_name": self.step_name,
            "timestamp": self.timestamp.isoformat(),
            "inputs": self._serialize_values(self.inputs),
            "outputs": self._serialize_values(self.outputs),
            "formula_used": self.formula_used,
            "references": self.references,
            "dependencies": self.dependencies,
            "status": self.status.value,
            "duration_ms": self.duration_ms,
            "notes": self.notes,
        }

    def _serialize_values(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize values, handling numpy and other types."""
        result = {}
        for key, value in data.items():
            if hasattr(value, 'tolist'):  # numpy array
                result[key] = value.tolist()
            elif hasattr(value, 'item'):  # numpy scalar
                result[key] = value.item()
            elif isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, Enum):
                result[key] = value.value
            else:
                result[key] = value
        return result

    def get_hash(self) -> str:
        """Calculate SHA-256 hash of this step."""
        content = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()


@dataclass
class ProvenanceRecord:
    """
    Complete provenance record for a calculation.

    Attributes:
        calculation_id: Unique identifier for this calculation
        calculation_type: Type of calculation performed
        created_at: When the calculation started
        completed_at: When the calculation completed
        steps: List of calculation steps
        final_results: Final output of the calculation
        provenance_hash: SHA-256 hash of the entire record
        metadata: Additional metadata
        version: Provenance format version
    """
    calculation_id: str
    calculation_type: str
    created_at: datetime
    completed_at: Optional[datetime]
    steps: List[CalculationStep]
    final_results: Dict[str, Any]
    provenance_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "calculation_id": self.calculation_id,
            "calculation_type": self.calculation_type,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "steps": [step.to_dict() for step in self.steps],
            "final_results": self.final_results,
            "provenance_hash": self.provenance_hash,
            "metadata": self.metadata,
            "version": self.version,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProvenanceRecord":
        """Create ProvenanceRecord from dictionary."""
        steps = [
            CalculationStep(
                step_id=s["step_id"],
                step_name=s["step_name"],
                timestamp=datetime.fromisoformat(s["timestamp"]),
                inputs=s["inputs"],
                outputs=s["outputs"],
                formula_used=s.get("formula_used"),
                references=s.get("references", []),
                dependencies=s.get("dependencies", []),
                status=CalculationStatus(s.get("status", "completed")),
                duration_ms=s.get("duration_ms", 0.0),
                notes=s.get("notes", ""),
            )
            for s in data.get("steps", [])
        ]

        return cls(
            calculation_id=data["calculation_id"],
            calculation_type=data["calculation_type"],
            created_at=datetime.fromisoformat(data["created_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            steps=steps,
            final_results=data.get("final_results", {}),
            provenance_hash=data.get("provenance_hash", ""),
            metadata=data.get("metadata", {}),
            version=data.get("version", "1.0.0"),
        )


# =============================================================================
# PROVENANCE BUILDER CLASS
# =============================================================================

class ProvenanceBuilder:
    """
    Builder class for creating provenance records.

    Provides a fluent interface for recording calculation steps
    and generating audit-ready provenance records.

    Example:
        >>> builder = ProvenanceBuilder("HX-RATING-001", "rating_calculation")
        >>> with builder.step("validate_inputs") as step:
        ...     step.add_input("tube_od", 0.019)
        ...     step.add_output("valid", True)
        >>> record = builder.finalize()
    """

    def __init__(
        self,
        calculation_id: Optional[str] = None,
        calculation_type: str = "unknown",
        level: ProvenanceLevel = ProvenanceLevel.STANDARD,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the ProvenanceBuilder.

        Args:
            calculation_id: Unique ID for this calculation (auto-generated if None)
            calculation_type: Type of calculation being performed
            level: Level of detail to record
            metadata: Additional metadata to include
        """
        self.calculation_id = calculation_id or str(uuid.uuid4())
        self.calculation_type = calculation_type
        self.level = level
        self.created_at = datetime.now(timezone.utc)
        self.steps: List[CalculationStep] = []
        self.metadata = metadata or {}
        self._current_step: Optional[StepBuilder] = None
        self._step_counter = 0

        logger.debug(f"ProvenanceBuilder initialized: {self.calculation_id}")

    def add_step(
        self,
        step_name: str,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        formula_used: Optional[str] = None,
        references: Optional[List[str]] = None,
        dependencies: Optional[List[str]] = None,
        notes: str = ""
    ) -> str:
        """
        Add a calculation step to the provenance record.

        Args:
            step_name: Name of the step
            inputs: Input data for the step
            outputs: Output data from the step
            formula_used: Formula or method used
            references: References to standards
            dependencies: Step IDs this step depends on
            notes: Additional notes

        Returns:
            Step ID of the added step
        """
        self._step_counter += 1
        step_id = f"step_{self._step_counter:04d}"

        step = CalculationStep(
            step_id=step_id,
            step_name=step_name,
            timestamp=datetime.now(timezone.utc),
            inputs=inputs or {},
            outputs=outputs or {},
            formula_used=formula_used,
            references=references or [],
            dependencies=dependencies or [],
            status=CalculationStatus.COMPLETED,
            notes=notes,
        )

        self.steps.append(step)
        logger.debug(f"Added step: {step_name} ({step_id})")

        return step_id

    def step(self, step_name: str) -> "StepBuilder":
        """
        Create a step builder context manager.

        Args:
            step_name: Name of the step

        Returns:
            StepBuilder context manager
        """
        self._step_counter += 1
        step_id = f"step_{self._step_counter:04d}"
        return StepBuilder(self, step_id, step_name)

    def add_metadata(self, key: str, value: Any) -> "ProvenanceBuilder":
        """
        Add metadata to the provenance record.

        Args:
            key: Metadata key
            value: Metadata value

        Returns:
            Self for method chaining
        """
        self.metadata[key] = value
        return self

    def _add_step_from_builder(self, step: CalculationStep) -> None:
        """Internal method to add a step from StepBuilder."""
        self.steps.append(step)
        logger.debug(f"Added step: {step.step_name} ({step.step_id})")

    def calculate_hash(self, data: Dict[str, Any]) -> str:
        """
        Calculate SHA-256 hash of data.

        Args:
            data: Data to hash

        Returns:
            Hexadecimal hash string
        """
        # Use canonical JSON for deterministic hashing
        content = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()

    def finalize(
        self,
        final_results: Optional[Dict[str, Any]] = None
    ) -> ProvenanceRecord:
        """
        Finalize and return the provenance record.

        Args:
            final_results: Final calculation results

        Returns:
            Complete ProvenanceRecord with hash
        """
        completed_at = datetime.now(timezone.utc)

        # Build preliminary record for hashing
        preliminary = {
            "calculation_id": self.calculation_id,
            "calculation_type": self.calculation_type,
            "created_at": self.created_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "steps": [s.to_dict() for s in self.steps],
            "final_results": final_results or {},
            "metadata": self.metadata,
        }

        # Calculate provenance hash
        provenance_hash = self.calculate_hash(preliminary)

        record = ProvenanceRecord(
            calculation_id=self.calculation_id,
            calculation_type=self.calculation_type,
            created_at=self.created_at,
            completed_at=completed_at,
            steps=self.steps,
            final_results=final_results or {},
            provenance_hash=provenance_hash,
            metadata=self.metadata,
        )

        logger.info(
            f"Provenance record finalized: {self.calculation_id} "
            f"({len(self.steps)} steps, hash: {provenance_hash[:16]}...)"
        )

        return record

    def verify_hash(self, record: ProvenanceRecord) -> bool:
        """
        Verify the integrity of a provenance record.

        Args:
            record: ProvenanceRecord to verify

        Returns:
            True if hash matches, False otherwise
        """
        # Reconstruct data for hashing
        data = {
            "calculation_id": record.calculation_id,
            "calculation_type": record.calculation_type,
            "created_at": record.created_at.isoformat(),
            "completed_at": record.completed_at.isoformat() if record.completed_at else None,
            "steps": [s.to_dict() for s in record.steps],
            "final_results": record.final_results,
            "metadata": record.metadata,
        }

        calculated_hash = self.calculate_hash(data)
        is_valid = calculated_hash == record.provenance_hash

        if not is_valid:
            logger.warning(
                f"Hash mismatch for {record.calculation_id}: "
                f"expected {record.provenance_hash[:16]}..., "
                f"got {calculated_hash[:16]}..."
            )

        return is_valid

    def get_dependency_chain(self, step_id: str) -> List[str]:
        """
        Get the chain of dependencies for a step.

        Args:
            step_id: Step ID to trace

        Returns:
            List of step IDs in dependency order
        """
        chain = []
        visited = set()

        def trace(sid: str) -> None:
            if sid in visited:
                return
            visited.add(sid)

            step = next((s for s in self.steps if s.step_id == sid), None)
            if step:
                for dep in step.dependencies:
                    trace(dep)
                chain.append(sid)

        trace(step_id)
        return chain


# =============================================================================
# STEP BUILDER CLASS (CONTEXT MANAGER)
# =============================================================================

class StepBuilder:
    """
    Context manager for building individual calculation steps.

    Provides a clean interface for adding inputs and outputs
    with automatic timing and status handling.

    Example:
        >>> with provenance.step("calculate_htc") as step:
        ...     step.add_input("reynolds", 50000)
        ...     step.set_formula("Dittus-Boelter: Nu = 0.023 * Re^0.8 * Pr^0.4")
        ...     h = calculate_htc(...)
        ...     step.add_output("htc", h)
    """

    def __init__(
        self,
        builder: ProvenanceBuilder,
        step_id: str,
        step_name: str
    ):
        """
        Initialize the StepBuilder.

        Args:
            builder: Parent ProvenanceBuilder
            step_id: Unique step identifier
            step_name: Human-readable step name
        """
        self._builder = builder
        self._step_id = step_id
        self._step_name = step_name
        self._start_time: Optional[datetime] = None
        self._inputs: Dict[str, Any] = {}
        self._outputs: Dict[str, Any] = {}
        self._formula: Optional[str] = None
        self._references: List[str] = []
        self._dependencies: List[str] = []
        self._notes: str = ""
        self._status = CalculationStatus.IN_PROGRESS

    def __enter__(self) -> "StepBuilder":
        """Enter the context manager."""
        self._start_time = datetime.now(timezone.utc)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit the context manager and record the step."""
        end_time = datetime.now(timezone.utc)
        duration_ms = (end_time - self._start_time).total_seconds() * 1000

        if exc_type is not None:
            self._status = CalculationStatus.FAILED
            self._notes += f"\nError: {exc_type.__name__}: {exc_val}"
        else:
            self._status = CalculationStatus.COMPLETED

        step = CalculationStep(
            step_id=self._step_id,
            step_name=self._step_name,
            timestamp=self._start_time,
            inputs=self._inputs,
            outputs=self._outputs,
            formula_used=self._formula,
            references=self._references,
            dependencies=self._dependencies,
            status=self._status,
            duration_ms=duration_ms,
            notes=self._notes,
        )

        self._builder._add_step_from_builder(step)

        # Don't suppress exceptions
        return False

    def add_input(self, name: str, value: Any) -> "StepBuilder":
        """
        Add an input value to the step.

        Args:
            name: Input parameter name
            value: Input value

        Returns:
            Self for method chaining
        """
        self._inputs[name] = value
        return self

    def add_inputs(self, inputs: Dict[str, Any]) -> "StepBuilder":
        """
        Add multiple input values to the step.

        Args:
            inputs: Dictionary of input parameters

        Returns:
            Self for method chaining
        """
        self._inputs.update(inputs)
        return self

    def add_output(self, name: str, value: Any) -> "StepBuilder":
        """
        Add an output value to the step.

        Args:
            name: Output parameter name
            value: Output value

        Returns:
            Self for method chaining
        """
        self._outputs[name] = value
        return self

    def add_outputs(self, outputs: Dict[str, Any]) -> "StepBuilder":
        """
        Add multiple output values to the step.

        Args:
            outputs: Dictionary of output parameters

        Returns:
            Self for method chaining
        """
        self._outputs.update(outputs)
        return self

    def set_formula(self, formula: str) -> "StepBuilder":
        """
        Set the formula used in this step.

        Args:
            formula: Formula or method description

        Returns:
            Self for method chaining
        """
        self._formula = formula
        return self

    def add_reference(self, reference: str) -> "StepBuilder":
        """
        Add a reference to standards or documentation.

        Args:
            reference: Reference string (e.g., "TEMA 9th Ed., Section 7")

        Returns:
            Self for method chaining
        """
        self._references.append(reference)
        return self

    def add_dependency(self, step_id: str) -> "StepBuilder":
        """
        Add a dependency on another step.

        Args:
            step_id: ID of the dependent step

        Returns:
            Self for method chaining
        """
        self._dependencies.append(step_id)
        return self

    def add_note(self, note: str) -> "StepBuilder":
        """
        Add a note or warning to the step.

        Args:
            note: Note text

        Returns:
            Self for method chaining
        """
        if self._notes:
            self._notes += f"\n{note}"
        else:
            self._notes = note
        return self

    def set_status(self, status: CalculationStatus) -> "StepBuilder":
        """
        Manually set the step status.

        Args:
            status: CalculationStatus value

        Returns:
            Self for method chaining
        """
        self._status = status
        return self

    @property
    def step_id(self) -> str:
        """Get the step ID."""
        return self._step_id


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_calculation_hash(
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    method: str = ""
) -> str:
    """
    Create a SHA-256 hash for a calculation.

    Args:
        inputs: Input parameters
        outputs: Output results
        method: Method or formula used

    Returns:
        Hexadecimal hash string
    """
    data = {
        "inputs": inputs,
        "outputs": outputs,
        "method": method,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    content = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(content.encode()).hexdigest()


def verify_provenance_record(record: Union[ProvenanceRecord, Dict[str, Any]]) -> bool:
    """
    Verify the integrity of a provenance record.

    Args:
        record: ProvenanceRecord or dictionary representation

    Returns:
        True if valid, False otherwise
    """
    if isinstance(record, dict):
        record = ProvenanceRecord.from_dict(record)

    builder = ProvenanceBuilder()
    return builder.verify_hash(record)


def load_provenance_from_json(json_string: str) -> ProvenanceRecord:
    """
    Load a provenance record from JSON string.

    Args:
        json_string: JSON representation of record

    Returns:
        ProvenanceRecord object
    """
    data = json.loads(json_string)
    return ProvenanceRecord.from_dict(data)


def save_provenance_to_file(record: ProvenanceRecord, filepath: str) -> None:
    """
    Save a provenance record to a file.

    Args:
        record: ProvenanceRecord to save
        filepath: Path to output file
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(record.to_json(indent=2))
    logger.info(f"Provenance record saved to: {filepath}")


def load_provenance_from_file(filepath: str) -> ProvenanceRecord:
    """
    Load a provenance record from a file.

    Args:
        filepath: Path to input file

    Returns:
        ProvenanceRecord object
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return load_provenance_from_json(f.read())


# Export all classes and functions
__all__ = [
    "CalculationStatus",
    "ProvenanceLevel",
    "CalculationStep",
    "ProvenanceRecord",
    "ProvenanceBuilder",
    "StepBuilder",
    "create_calculation_hash",
    "verify_provenance_record",
    "load_provenance_from_json",
    "save_provenance_to_file",
    "load_provenance_from_file",
]
