# -*- coding: utf-8 -*-
"""
Provenance Tracking Framework for GreenLang Agents.

This module provides comprehensive data lineage and provenance tracking
capabilities for ensuring traceability, audit compliance, and reproducibility
of all agent computations and data transformations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
import uuid
import hashlib
import json
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class ProvenanceAction(str, Enum):
    """Types of provenance actions."""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    TRANSFORM = "transform"
    VALIDATE = "validate"
    AGGREGATE = "aggregate"
    DERIVE = "derive"
    IMPORT = "import"
    EXPORT = "export"


class DataSourceType(str, Enum):
    """Types of data sources."""
    SENSOR = "sensor"
    DATABASE = "database"
    API = "api"
    FILE = "file"
    USER_INPUT = "user_input"
    CALCULATION = "calculation"
    EXTERNAL_SYSTEM = "external_system"
    CACHE = "cache"


@dataclass
class DataLineage:
    """
    Data lineage record for tracking data origins.

    Attributes:
        lineage_id: Unique identifier for the lineage
        source_type: Type of data source
        source_id: Identifier of the source
        source_name: Human-readable source name
        timestamp: When the data was sourced
        metadata: Additional lineage metadata
        parent_lineages: Parent lineage IDs (for derived data)
        transformations: List of transformations applied
    """
    lineage_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_type: DataSourceType = DataSourceType.CALCULATION
    source_id: str = ""
    source_name: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_lineages: List[str] = field(default_factory=list)
    transformations: List[str] = field(default_factory=list)
    checksum: Optional[str] = None

    def add_parent(self, parent_id: str):
        """Add a parent lineage."""
        if parent_id not in self.parent_lineages:
            self.parent_lineages.append(parent_id)

    def add_transformation(self, transformation: str):
        """Add a transformation description."""
        self.transformations.append(transformation)

    def compute_checksum(self, data: Any) -> str:
        """Compute checksum for data verification."""
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data, sort_keys=True, default=str)
        else:
            data_str = str(data)
        self.checksum = hashlib.sha256(data_str.encode()).hexdigest()
        return self.checksum

    def to_dict(self) -> Dict[str, Any]:
        """Convert lineage to dictionary."""
        return {
            "lineage_id": self.lineage_id,
            "source_type": self.source_type.value,
            "source_id": self.source_id,
            "source_name": self.source_name,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "parent_lineages": self.parent_lineages,
            "transformations": self.transformations,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataLineage":
        """Create lineage from dictionary."""
        return cls(
            lineage_id=data.get("lineage_id", str(uuid.uuid4())),
            source_type=DataSourceType(data.get("source_type", "calculation")),
            source_id=data.get("source_id", ""),
            source_name=data.get("source_name", ""),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.utcnow(),
            metadata=data.get("metadata", {}),
            parent_lineages=data.get("parent_lineages", []),
            transformations=data.get("transformations", []),
            checksum=data.get("checksum"),
        )


@dataclass
class ProvenanceRecord:
    """
    Single provenance record for an action.

    Attributes:
        record_id: Unique identifier for the record
        action: Type of action performed
        agent_id: ID of the agent performing the action
        timestamp: When the action occurred
        inputs: Input data references
        outputs: Output data references
        parameters: Parameters used in the action
        lineage: Data lineage information
        duration_ms: Duration of the action in milliseconds
        success: Whether the action succeeded
        error: Error message if action failed
        metadata: Additional record metadata
    """
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    action: ProvenanceAction = ProvenanceAction.TRANSFORM
    agent_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    lineage: Optional[DataLineage] = None
    duration_ms: Optional[float] = None
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        return {
            "record_id": self.record_id,
            "action": self.action.value,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp.isoformat(),
            "inputs": self.inputs,
            "outputs": self.outputs,
            "parameters": self.parameters,
            "lineage": self.lineage.to_dict() if self.lineage else None,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProvenanceRecord":
        """Create record from dictionary."""
        lineage_data = data.get("lineage")
        lineage = DataLineage.from_dict(lineage_data) if lineage_data else None

        return cls(
            record_id=data.get("record_id", str(uuid.uuid4())),
            action=ProvenanceAction(data.get("action", "transform")),
            agent_id=data.get("agent_id", ""),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.utcnow(),
            inputs=data.get("inputs", {}),
            outputs=data.get("outputs", {}),
            parameters=data.get("parameters", {}),
            lineage=lineage,
            duration_ms=data.get("duration_ms"),
            success=data.get("success", True),
            error=data.get("error"),
            metadata=data.get("metadata", {}),
        )


class ProvenanceTracker:
    """
    Provenance tracker for recording and querying data lineage.

    This class provides comprehensive tracking of all data transformations,
    calculations, and agent actions for audit compliance and reproducibility.

    Example:
        >>> tracker = ProvenanceTracker("GL-006")
        >>> with tracker.track_operation("calculate_heat_recovery"):
        ...     result = calculate_heat_recovery(data)
        >>> tracker.get_lineage(result_id)
    """

    def __init__(self, agent_id: str, max_records: int = 10000):
        """
        Initialize the provenance tracker.

        Args:
            agent_id: ID of the agent using this tracker
            max_records: Maximum records to keep in memory
        """
        self.agent_id = agent_id
        self.max_records = max_records
        self._records: List[ProvenanceRecord] = []
        self._lineages: Dict[str, DataLineage] = {}
        self._current_record: Optional[ProvenanceRecord] = None
        self._logger = logging.getLogger(f"provenance.{agent_id}")

    def create_lineage(
        self,
        source_type: DataSourceType,
        source_id: str,
        source_name: str = "",
        data: Any = None,
        parent_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DataLineage:
        """
        Create a new data lineage record.

        Args:
            source_type: Type of data source
            source_id: Identifier of the source
            source_name: Human-readable source name
            data: Data to track (for checksum calculation)
            parent_ids: Parent lineage IDs
            metadata: Additional metadata

        Returns:
            Created DataLineage object
        """
        lineage = DataLineage(
            source_type=source_type,
            source_id=source_id,
            source_name=source_name,
            metadata=metadata or {},
            parent_lineages=parent_ids or [],
        )

        if data is not None:
            lineage.compute_checksum(data)

        self._lineages[lineage.lineage_id] = lineage
        self._logger.debug(f"Created lineage: {lineage.lineage_id}")
        return lineage

    def record_action(
        self,
        action: ProvenanceAction,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        lineage: Optional[DataLineage] = None,
        duration_ms: Optional[float] = None,
        success: bool = True,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProvenanceRecord:
        """
        Record a provenance action.

        Args:
            action: Type of action performed
            inputs: Input data references
            outputs: Output data references
            parameters: Parameters used
            lineage: Associated data lineage
            duration_ms: Duration in milliseconds
            success: Whether action succeeded
            error: Error message if failed
            metadata: Additional metadata

        Returns:
            Created ProvenanceRecord
        """
        record = ProvenanceRecord(
            action=action,
            agent_id=self.agent_id,
            inputs=inputs or {},
            outputs=outputs or {},
            parameters=parameters or {},
            lineage=lineage,
            duration_ms=duration_ms,
            success=success,
            error=error,
            metadata=metadata or {},
        )

        self._records.append(record)

        # Enforce max records limit
        if len(self._records) > self.max_records:
            self._records = self._records[-self.max_records:]

        self._logger.debug(f"Recorded action: {action.value} - {record.record_id}")
        return record

    @contextmanager
    def track_operation(
        self,
        operation_name: str,
        action: ProvenanceAction = ProvenanceAction.TRANSFORM,
        inputs: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        """
        Context manager for tracking an operation.

        Args:
            operation_name: Name of the operation
            action: Type of action
            inputs: Input data
            parameters: Operation parameters

        Yields:
            ProvenanceRecord being built
        """
        start_time = datetime.utcnow()
        record = ProvenanceRecord(
            action=action,
            agent_id=self.agent_id,
            inputs=inputs or {},
            parameters=parameters or {},
            metadata={"operation_name": operation_name},
        )
        self._current_record = record

        try:
            yield record
            record.success = True
        except Exception as e:
            record.success = False
            record.error = str(e)
            raise
        finally:
            end_time = datetime.utcnow()
            record.duration_ms = (end_time - start_time).total_seconds() * 1000
            self._records.append(record)
            self._current_record = None

    def get_lineage(self, lineage_id: str) -> Optional[DataLineage]:
        """Get a lineage record by ID."""
        return self._lineages.get(lineage_id)

    def get_record(self, record_id: str) -> Optional[ProvenanceRecord]:
        """Get a provenance record by ID."""
        for record in self._records:
            if record.record_id == record_id:
                return record
        return None

    def get_records_by_action(self, action: ProvenanceAction) -> List[ProvenanceRecord]:
        """Get all records for a specific action type."""
        return [r for r in self._records if r.action == action]

    def get_records_in_range(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> List[ProvenanceRecord]:
        """Get all records within a time range."""
        return [
            r for r in self._records
            if start_time <= r.timestamp <= end_time
        ]

    def get_failed_records(self) -> List[ProvenanceRecord]:
        """Get all failed records."""
        return [r for r in self._records if not r.success]

    def get_full_lineage_chain(self, lineage_id: str) -> List[DataLineage]:
        """
        Get the full lineage chain for a data point.

        Args:
            lineage_id: ID of the lineage to trace

        Returns:
            List of lineages from root to the specified lineage
        """
        chain = []
        visited: Set[str] = set()

        def _trace(lid: str):
            if lid in visited:
                return
            visited.add(lid)

            lineage = self._lineages.get(lid)
            if lineage:
                for parent_id in lineage.parent_lineages:
                    _trace(parent_id)
                chain.append(lineage)

        _trace(lineage_id)
        return chain

    def export_records(self) -> List[Dict[str, Any]]:
        """Export all records as dictionaries."""
        return [r.to_dict() for r in self._records]

    def export_lineages(self) -> Dict[str, Dict[str, Any]]:
        """Export all lineages as dictionaries."""
        return {lid: l.to_dict() for lid, l in self._lineages.items()}

    def clear(self):
        """Clear all records and lineages."""
        self._records = []
        self._lineages = {}
        self._logger.info("Cleared all provenance data")

    def get_statistics(self) -> Dict[str, Any]:
        """Get provenance statistics."""
        action_counts = {}
        for record in self._records:
            action = record.action.value
            action_counts[action] = action_counts.get(action, 0) + 1

        success_count = sum(1 for r in self._records if r.success)
        failure_count = len(self._records) - success_count

        total_duration = sum(r.duration_ms or 0 for r in self._records)

        return {
            "total_records": len(self._records),
            "total_lineages": len(self._lineages),
            "action_counts": action_counts,
            "success_count": success_count,
            "failure_count": failure_count,
            "total_duration_ms": total_duration,
            "average_duration_ms": total_duration / len(self._records) if self._records else 0,
        }


__all__ = [
    'ProvenanceAction',
    'DataSourceType',
    'DataLineage',
    'ProvenanceRecord',
    'ProvenanceTracker',
]
