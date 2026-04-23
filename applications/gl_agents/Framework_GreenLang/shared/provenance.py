"""
GreenLang Framework - Provenance Tracking

SHA-256 based provenance tracking for deterministic calculations.
Ensures all calculations are reproducible and auditable.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
import hashlib
import json
import uuid


@dataclass
class ProvenanceRecord:
    """Record of a single computation's provenance."""

    record_id: str
    computation_type: str
    inputs_hash: str
    outputs_hash: str
    computation_hash: str
    agent_id: str
    agent_version: str
    timestamp: datetime
    execution_time_ms: float
    parameters: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "record_id": self.record_id,
            "computation_type": self.computation_type,
            "inputs_hash": self.inputs_hash,
            "outputs_hash": self.outputs_hash,
            "computation_hash": self.computation_hash,
            "agent_id": self.agent_id,
            "agent_version": self.agent_version,
            "timestamp": self.timestamp.isoformat(),
            "execution_time_ms": self.execution_time_ms,
            "parameters": self.parameters,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProvenanceRecord":
        """Create from dictionary."""
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class ProvenanceTracker:
    """
    Provenance tracker for deterministic calculations.

    Provides SHA-256 hashing for:
    - Input data
    - Output results
    - Combined computation hash

    Usage:
        >>> tracker = ProvenanceTracker(agent_id="GL-006", version="1.0.0")
        >>> with tracker.track("pinch_analysis", inputs) as ctx:
        ...     result = calculate_pinch(inputs)
        ...     ctx.set_output(result)
        >>> print(tracker.last_record.computation_hash)
    """

    def __init__(
        self,
        agent_id: str,
        version: str,
        store_records: bool = True,
    ):
        """
        Initialize provenance tracker.

        Args:
            agent_id: Agent identifier (e.g., "GL-006")
            version: Agent version (e.g., "1.0.0")
            store_records: Whether to store records in memory
        """
        self.agent_id = agent_id
        self.version = version
        self.store_records = store_records
        self._records: List[ProvenanceRecord] = []
        self._last_record: Optional[ProvenanceRecord] = None

    @property
    def last_record(self) -> Optional[ProvenanceRecord]:
        """Get the last provenance record."""
        return self._last_record

    @property
    def records(self) -> List[ProvenanceRecord]:
        """Get all stored records."""
        return self._records.copy()

    def compute_hash(
        self,
        data: Any,
        include_types: bool = True,
    ) -> str:
        """
        Compute SHA-256 hash of data.

        Args:
            data: Data to hash (will be JSON serialized)
            include_types: Include type information in hash

        Returns:
            64-character hex hash string
        """
        if include_types:
            hashable = self._make_hashable_with_types(data)
        else:
            hashable = self._make_hashable(data)

        json_str = json.dumps(hashable, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()

    def create_record(
        self,
        computation_type: str,
        inputs: Any,
        outputs: Any,
        execution_time_ms: float,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProvenanceRecord:
        """
        Create a provenance record for a computation.

        Args:
            computation_type: Type of computation (e.g., "pinch_analysis")
            inputs: Input data
            outputs: Output results
            execution_time_ms: Execution time in milliseconds
            parameters: Computation parameters
            metadata: Additional metadata

        Returns:
            ProvenanceRecord with all hashes computed
        """
        inputs_hash = self.compute_hash(inputs)
        outputs_hash = self.compute_hash(outputs)

        # Combined hash includes inputs, outputs, agent info, and parameters
        combined_data = {
            "inputs_hash": inputs_hash,
            "outputs_hash": outputs_hash,
            "agent_id": self.agent_id,
            "agent_version": self.version,
            "computation_type": computation_type,
            "parameters": parameters or {},
        }
        computation_hash = self.compute_hash(combined_data)

        record = ProvenanceRecord(
            record_id=str(uuid.uuid4()),
            computation_type=computation_type,
            inputs_hash=inputs_hash,
            outputs_hash=outputs_hash,
            computation_hash=computation_hash,
            agent_id=self.agent_id,
            agent_version=self.version,
            timestamp=datetime.now(timezone.utc),
            execution_time_ms=execution_time_ms,
            parameters=parameters or {},
            metadata=metadata or {},
        )

        self._last_record = record
        if self.store_records:
            self._records.append(record)

        return record

    def verify_hash(
        self,
        data: Any,
        expected_hash: str,
    ) -> bool:
        """
        Verify data matches expected hash.

        Args:
            data: Data to verify
            expected_hash: Expected hash value

        Returns:
            True if hash matches
        """
        actual_hash = self.compute_hash(data)
        return actual_hash == expected_hash

    def track(
        self,
        computation_type: str,
        inputs: Any,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> "TrackingContext":
        """
        Context manager for tracking computations.

        Usage:
            >>> with tracker.track("calculation", inputs) as ctx:
            ...     result = do_calculation(inputs)
            ...     ctx.set_output(result)
        """
        return TrackingContext(
            tracker=self,
            computation_type=computation_type,
            inputs=inputs,
            parameters=parameters,
        )

    def get_audit_trail(
        self,
        computation_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[ProvenanceRecord]:
        """
        Get audit trail of computations.

        Args:
            computation_type: Filter by type
            start_time: Filter by start time
            end_time: Filter by end time

        Returns:
            List of matching provenance records
        """
        records = self._records

        if computation_type:
            records = [r for r in records if r.computation_type == computation_type]

        if start_time:
            records = [r for r in records if r.timestamp >= start_time]

        if end_time:
            records = [r for r in records if r.timestamp <= end_time]

        return records

    def export_records(self, format: str = "json") -> str:
        """Export records to JSON format."""
        records_data = [r.to_dict() for r in self._records]
        return json.dumps(records_data, indent=2, default=str)

    def _make_hashable(self, obj: Any) -> Any:
        """Convert object to hashable representation."""
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [self._make_hashable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._make_hashable(v) for k, v in sorted(obj.items())}
        elif hasattr(obj, '__dict__'):
            return self._make_hashable(obj.__dict__)
        elif hasattr(obj, 'dict'):
            return self._make_hashable(obj.dict())
        else:
            return str(obj)

    def _make_hashable_with_types(self, obj: Any) -> Any:
        """Convert object to hashable with type information."""
        if obj is None:
            return {"__type__": "null", "__value__": None}
        elif isinstance(obj, bool):
            return {"__type__": "bool", "__value__": obj}
        elif isinstance(obj, int):
            return {"__type__": "int", "__value__": obj}
        elif isinstance(obj, float):
            return {"__type__": "float", "__value__": round(obj, 10)}
        elif isinstance(obj, str):
            return {"__type__": "str", "__value__": obj}
        elif isinstance(obj, (list, tuple)):
            return {
                "__type__": type(obj).__name__,
                "__value__": [self._make_hashable_with_types(item) for item in obj]
            }
        elif isinstance(obj, dict):
            return {
                "__type__": "dict",
                "__value__": {
                    k: self._make_hashable_with_types(v)
                    for k, v in sorted(obj.items())
                }
            }
        else:
            return {"__type__": type(obj).__name__, "__value__": str(obj)}


class TrackingContext:
    """Context manager for provenance tracking."""

    def __init__(
        self,
        tracker: ProvenanceTracker,
        computation_type: str,
        inputs: Any,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        self.tracker = tracker
        self.computation_type = computation_type
        self.inputs = inputs
        self.parameters = parameters
        self.outputs: Any = None
        self.metadata: Dict[str, Any] = {}
        self._start_time: Optional[datetime] = None
        self._record: Optional[ProvenanceRecord] = None

    def set_output(self, outputs: Any) -> None:
        """Set the computation outputs."""
        self.outputs = outputs

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the record."""
        self.metadata[key] = value

    @property
    def record(self) -> Optional[ProvenanceRecord]:
        """Get the created record."""
        return self._record

    def __enter__(self) -> "TrackingContext":
        """Enter tracking context."""
        self._start_time = datetime.now(timezone.utc)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit tracking context and create record."""
        if exc_type is None and self.outputs is not None:
            end_time = datetime.now(timezone.utc)
            execution_time_ms = (end_time - self._start_time).total_seconds() * 1000

            self._record = self.tracker.create_record(
                computation_type=self.computation_type,
                inputs=self.inputs,
                outputs=self.outputs,
                execution_time_ms=execution_time_ms,
                parameters=self.parameters,
                metadata=self.metadata,
            )
